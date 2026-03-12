"""
Stochastic MCMC calibration with DETERMINISTIC z_scale.

Key differences from full stochastic version:
1. z_scale is FIXED after mean-fit (not calibrated stochastically)
2. Only calibrates: θ = [μ_η, μ_α, σ_η, σ_α]  (4 params instead of 6)
3. Faster, more interpretable, physically justified

Fixes vs previous version:
- Correlation penalty in objective forces 2D cloud (not 1D line)
- Wider bounds on std_alpha to allow larger spread
- Larger proposal scales for std_eta / std_alpha
- Observation noise applied once (not twice) in generate_predictions
- Initial x0 uses larger std values so MCMC starts away from lower bounds
- Removed .sampling import (seed_from_x inlined to avoid JAX dependency)
- Added calibrate = calibrate_mcmc alias for backwards compatibility
"""

import numpy as np
from scipy.stats import gaussian_kde


def _seed_from_x(x: np.ndarray, salt: int = 0) -> int:
    """Deterministic integer seed from a parameter vector."""
    x = np.asarray(x, dtype=float)
    x_int = np.round(x * 1e6).astype(np.int64)
    w = np.arange(1, len(x_int) + 1, dtype=np.int64)
    s = int(np.sum(x_int * w)) + int(salt)
    return int(np.mod(s, 2**32 - 1))


class StochasticCalibrationMCMC:
    """
    Bayesian stochastic calibration with DETERMINISTIC z_scale.

    Calibrates only η and α as random variables.
    z_scale is fixed from deterministic mean-fit.

    Hyperparameters: [μ_η, μ_α, σ_η, σ_α]  (4 total)
    """

    def __init__(self, exp_widths_um, exp_depths_um, z_scale):
        self.exp_widths = np.asarray(exp_widths_um, dtype=float)
        self.exp_depths = np.asarray(exp_depths_um, dtype=float)
        self.z_scale    = float(z_scale)

        self.exp_w_mean = float(np.mean(self.exp_widths))
        self.exp_w_std  = float(np.std(self.exp_widths,  ddof=1))
        self.exp_d_mean = float(np.mean(self.exp_depths))
        self.exp_d_std  = float(np.std(self.exp_depths,  ddof=1))

        if len(self.exp_widths) > 5:
            self.exp_corr = float(np.corrcoef(self.exp_widths, self.exp_depths)[0, 1])
        else:
            self.exp_corr = 0.0

        self.mu_star          = None
        self.sigma_star       = None
        self.mcmc_chain       = None
        self.mcmc_diagnostics = None

    # ------------------------------------------------------------------
    def sample_parameters(self, mu, sigma, n_samples, seed=None):
        rng = np.random.default_rng(seed)
        samples = np.zeros((n_samples, 3), dtype=float)
        samples[:, 0] = rng.normal(mu[0], sigma[0], size=n_samples)
        samples[:, 1] = rng.normal(mu[1], sigma[1], size=n_samples)
        samples[:, 2] = self.z_scale
        samples[:, 0] = np.clip(samples[:, 0], 0.15, 1.2)
        samples[:, 1] = np.clip(samples[:, 1], 0.6,  1.6)
        return samples

    def run_ensemble(self, melt_dims_fn, sigma_m, param_samples):
        widths_um, depths_um = melt_dims_fn(param_samples, float(sigma_m))
        w = np.asarray(widths_um, dtype=float)
        d = np.asarray(depths_um, dtype=float)
        valid = (w > 0) & (d > 0) & np.isfinite(w) & np.isfinite(d)
        return w[valid], d[valid]

    # ------------------------------------------------------------------
    def kl_divergence_kde(self, p_samples, q_samples, n_grid=300, seed=None):
        p_samples = np.asarray(p_samples, dtype=float)
        q_samples = np.asarray(q_samples, dtype=float)
        if len(p_samples) < 5 or len(q_samples) < 5:
            return 1e6
        try:
            rng   = np.random.default_rng(seed if seed is not None else 0)
            p_jit = p_samples + rng.normal(0, 0.1, len(p_samples))
            q_jit = q_samples + rng.normal(0, 0.1, len(q_samples))
            p_kde = gaussian_kde(p_jit)
            q_kde = gaussian_kde(q_jit)
            lo   = min(p_samples.min(), q_samples.min())
            hi   = max(p_samples.max(), q_samples.max())
            span = hi - lo + 1e-12
            grid = np.linspace(lo - 0.15*span, hi + 0.15*span, n_grid)
            p_vals = np.maximum(p_kde(grid), 1e-10)
            q_vals = np.maximum(q_kde(grid), 1e-10)
            p_vals /= (np.trapz(p_vals, grid) + 1e-12)
            q_vals /= (np.trapz(q_vals, grid) + 1e-12)
            kl = np.trapz(p_vals * np.log(p_vals / q_vals), grid)
            return float(max(0.0, kl))
        except Exception:
            return 1e6

    # ------------------------------------------------------------------
    def objective_function(
        self,
        hyperparams,
        melt_dims_fn,
        sigma_m,
        n_samples  = 400,
        w_mean     = 2.0,
        w_std      = 5.0,
        w_kl       = 1.1,
        w_corr     = 5.0,
        reg_lambda = 1e-4,
        seed       = None,
    ):
        hyperparams = np.asarray(hyperparams, dtype=float)
        mu    = hyperparams[:2]
        sigma = hyperparams[2:4]

        if np.any(sigma <= 0) or np.any(sigma > 0.3):
            return 1e9

        if seed is None:
            seed = _seed_from_x(hyperparams)

        param_samples = self.sample_parameters(mu, sigma, n_samples, seed=seed)
        sim_w, sim_d  = self.run_ensemble(melt_dims_fn, sigma_m, param_samples)

        if len(sim_w) < 5:
            return 1e6

        mean_loss = (
            ((np.mean(sim_w) - self.exp_w_mean) / (self.exp_w_mean + 1e-12)) ** 2 +
            ((np.mean(sim_d) - self.exp_d_mean) / (self.exp_d_mean + 1e-12)) ** 2
        )
        std_loss = (
            ((np.std(sim_w, ddof=1) - self.exp_w_std) / (self.exp_w_std + 1e-12)) ** 2 +
            ((np.std(sim_d, ddof=1) - self.exp_d_std) / (self.exp_d_std + 1e-12)) ** 2
        )
        kl_w    = self.kl_divergence_kde(self.exp_widths, sim_w, seed=seed)
        kl_d    = self.kl_divergence_kde(self.exp_depths, sim_d, seed=seed + 1)
        kl_loss = kl_w + kl_d

        corr_loss = 0.0
        if len(sim_w) > 5 and np.std(sim_w) > 0 and np.std(sim_d) > 0:
            sim_corr  = float(np.corrcoef(sim_w, sim_d)[0, 1])
            corr_loss = (sim_corr - self.exp_corr) ** 2

        reg = float(np.sum(sigma ** 2))

        return float(
            w_mean * mean_loss + w_std * std_loss +
            w_kl * kl_loss + w_corr * corr_loss +
            reg_lambda * reg
        )

    # ------------------------------------------------------------------
    def _log_prior_uniform(self, hyperparams, bounds):
        for val, (lo, hi) in zip(hyperparams, bounds):
            if val < lo or val > hi:
                return -np.inf
        return 0.0

    def _log_likelihood(self, hyperparams, melt_dims_fn, sigma_m, n_samples, beta, seed):
        J = self.objective_function(hyperparams, melt_dims_fn, sigma_m,
                                    n_samples=n_samples, seed=seed)
        if not np.isfinite(J) or J > 1e12:
            return -np.inf
        return -beta * J

    def _log_posterior(self, hyperparams, bounds, melt_dims_fn, sigma_m,
                       n_samples, beta, seed):
        lp = self._log_prior_uniform(hyperparams, bounds)
        if not np.isfinite(lp):
            return -np.inf
        ll = self._log_likelihood(hyperparams, melt_dims_fn, sigma_m,
                                   n_samples, beta, seed)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll

    # ------------------------------------------------------------------
    def calibrate_mcmc(
        self,
        melt_dims_fn,
        sigma_m,
        init_from_meanfit = None,
        n_steps           = 2000,
        burn_in           = 400,
        n_samples         = 80,
        beta              = 1.0,
        proposal_scales   = None,
        verbose           = True,
        seed              = 123,
    ):
        if verbose:
            print("=" * 70)
            print("Stochastic Calibration via MCMC (Fixed z_scale)")
            print("=" * 70)
            print(f"z_scale fixed at   : {self.z_scale:.4f}")
            print(f"Experimental data  : {len(self.exp_widths)} width / "
                  f"{len(self.exp_depths)} depth measurements")
            print(f"Experimental corr  : {self.exp_corr:.3f}")
            print(f"MCMC settings      : {n_steps} steps, {burn_in} burn-in, "
                  f"ensemble size {n_samples}")

        rng = np.random.default_rng(seed)

        bounds = [
            (0.45, 0.9),    # mu_eta   — empirically calibrated for IN718
            (0.6, 1.2),    # mu_alpha
            (0.005, 0.06),   # std_eta
            (0.015, 0.1),   # std_alpha
        ]

        if init_from_meanfit is not None:
            x0 = np.array([
                init_from_meanfit['eta'],
                init_from_meanfit['alpha_mult'],
                0.025,   # std_eta initial
                0.045,   # std_alpha initial
            ], dtype=float)
        else:
            x0 = np.array([0.60, 0.95, 0.025, 0.045], dtype=float)

        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])

        if proposal_scales is None:
            # These scales gave ESS 103-191 at 7500 steps in empirical testing.
            # Small enough for decent acceptance (~50%), large enough for mixing.
            proposal_scales = np.array([0.004, 0.006, 0.005, 0.008], dtype=float)

        def propose(x):
            return x + rng.normal(0.0, proposal_scales, size=x.shape)

        x_cur    = x0.copy()
        logp_cur = self._log_posterior(x_cur, bounds, melt_dims_fn, sigma_m,
                                       n_samples, beta, seed)

        if not np.isfinite(logp_cur):
            raise ValueError(f"Initial point has -inf posterior. x0={x0}.")

        chain    = np.zeros((n_steps, 4))
        logps    = np.full(n_steps, -np.inf)
        accepted = np.zeros(n_steps, dtype=int)

        for t in range(n_steps):
            x_prop = propose(x_cur)
            if not all(b[0] <= x <= b[1] for x, b in zip(x_prop, bounds)):
                chain[t]  = x_cur
                logps[t]  = logp_cur
                continue

            logp_prop = self._log_posterior(x_prop, bounds, melt_dims_fn, sigma_m,
                                            n_samples, beta, seed + t + 1)

            if np.log(rng.uniform()) < (logp_prop - logp_cur):
                x_cur    = x_prop
                logp_cur = logp_prop
                accepted[t] = 1

            chain[t] = x_cur
            logps[t] = logp_cur

            if verbose and (t + 1) % 200 == 0:
                ar = accepted[: t + 1].mean()
                print(f"  Step {t+1:4d}/{n_steps}  accept={ar:.3f}  "
                      f"logp={logp_cur:.3f}  "
                      f"μ_η={x_cur[0]:.4f}  μ_α={x_cur[1]:.4f}  "
                      f"σ_η={x_cur[2]:.4f}  σ_α={x_cur[3]:.4f}")

        post_chain = chain[burn_in:]
        post_logps = logps[burn_in:]
        map_idx    = int(np.argmax(post_logps))
        x_map      = post_chain[map_idx]
        x_mean     = post_chain.mean(axis=0)
        x_std      = post_chain.std(axis=0)

        self.mu_star    = x_map[:2]
        self.sigma_star = x_map[2:4]
        self.mcmc_chain = chain
        self.mcmc_diagnostics = {
            'accept_rate': float(accepted.mean()),
            'burn_in':     burn_in,
            'n_effective': len(post_chain),
        }

        if verbose:
            print("\n" + "=" * 70)
            print("MCMC Complete (Fixed z_scale)")
            print("=" * 70)
            print(f"Acceptance rate : {accepted.mean():.3f}")
            print(f"\nMAP estimate (z_scale = {self.z_scale:.4f} fixed):")
            print(f"  μ_η  = {x_map[0]:.4f}   μ_α  = {x_map[1]:.4f}")
            print(f"  σ_η  = {x_map[2]:.4f}   σ_α  = {x_map[3]:.4f}")
            print("\nPosterior mean ± std:")
            for name, m, s in zip(['μ_η', 'μ_α', 'σ_η', 'σ_α'], x_mean, x_std):
                print(f"  {name}: {m:.4f} ± {s:.4f}")

        return {
            'chain':           chain,
            'logp':            logps,
            'accepted':        accepted,
            'accept_rate':     float(accepted.mean()),
            'burn_in':         burn_in,
            'posterior_chain': post_chain,        # used by agent for ESS
            'posterior_logp':  post_logps,
            'x_map':           x_map,
            'x_mean':          x_mean,
            'x_std':           x_std,
            'bounds':          bounds,
            'proposal_scales': proposal_scales,
            'beta':            beta,
            'n_effective':     len(post_chain),
            'std_bounds_used': {
                'std_eta':   bounds[2],
                'std_alpha': bounds[3],
            },
        }

    # Backwards-compatibility alias
    calibrate = calibrate_mcmc

    # ------------------------------------------------------------------
    def generate_predictions(
        self,
        melt_dims_fn,
        sigma_m,
        n_samples     = 800,
        seed          = 2026,
        use_map       = True,
        add_obs_noise = True,
    ):
        if self.mu_star is None:
            raise RuntimeError("Must run calibrate_mcmc() first.")

        param_samples = self.sample_parameters(
            self.mu_star, self.sigma_star, n_samples, seed=seed
        )
        widths_um, depths_um = self.run_ensemble(melt_dims_fn, sigma_m, param_samples)

        if add_obs_noise:
            rng = np.random.default_rng(seed)
            widths_um = widths_um + rng.normal(0, self.exp_w_std, size=len(widths_um))
            depths_um = depths_um + rng.normal(0, self.exp_d_std, size=len(depths_um))

        return widths_um, depths_um, param_samples[:len(widths_um)]
