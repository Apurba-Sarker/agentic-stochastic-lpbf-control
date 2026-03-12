"""
Goldak-style Green's function thermal model in JAX.
Returns melt_dims(params (N,3), sigma_m) -> (widths_um (N,), depths_um (N,))
params: [eta, alpha_mult, z_scale]
"""
import numpy as np
from .jax_utils import enable_x64


def _get_k(T):
    return 0.016 * T + 6.39


def make_melt_dims_jax(P_W, V_mmps, material_props,
                        track_len=3e-3, dx=5e-6, n_scan=60,
                        y_max=300e-6, z_min=-300e-6, check_x_offset=40e-6):
    enable_x64()
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap, lax

    RHO    = float(material_props.RHO)
    CP     = float(material_props.CP)
    T_MELT = float(material_props.T_MELT)
    T0     = float(material_props.T0)
    V_mps  = float(V_mmps) * 1e-3
    P      = float(P_W)

    n_src   = int(track_len / dx)
    dt_step = dx / V_mps
    i       = jnp.arange(n_src, dtype=jnp.float64)
    x_i     = i * dx
    t_i     = i * dt_step
    dt_arr  = (t_i[-1] + 1e-9) - t_i
    dx_sq   = ((x_i[-1] - check_x_offset) - x_i) ** 2

    y_scan = jnp.linspace(0.0, y_max, n_scan)
    y2     = y_scan ** 2
    z_scan = jnp.linspace(0.0, z_min, n_scan)
    z2     = z_scan ** 2

    base_alpha = _get_k(T_MELT) / (RHO * CP)
    pref_const = (6.0 * jnp.sqrt(3.0) * P) / (RHO * CP * jnp.pi * jnp.sqrt(jnp.pi))
    eps = 1e-12

    def _width(Ty):
        above = Ty >= T_MELT
        def melted(_):
            cross = above[:-1] & (~above[1:])
            has_cross = jnp.any(cross)
            idx = jnp.argmax(cross)
            idx_last = jnp.max(jnp.where(above, jnp.arange(above.size), -1))
            def do_cross(_):
                frac = (Ty[idx]-T_MELT)/(Ty[idx]-Ty[idx+1]+eps)
                return 2.0*(y_scan[idx]+frac*(y_scan[idx+1]-y_scan[idx]))
            def do_nocross(_):
                return 2.0*y_scan[idx_last]
            return lax.cond(has_cross, do_cross, do_nocross, None)
        return lax.cond(jnp.max(Ty)<T_MELT, lambda _:0.0, melted, None)

    def _depth(Tz):
        above = Tz >= T_MELT
        def melted(_):
            cross = above[:-1] & (~above[1:])
            has_cross = jnp.any(cross)
            idx = jnp.argmax(cross)
            idx_last = jnp.max(jnp.where(above, jnp.arange(above.size), -1))
            def do_cross(_):
                frac = (Tz[idx]-T_MELT)/(Tz[idx]-Tz[idx+1]+eps)
                return jnp.abs(z_scan[idx]+frac*(z_scan[idx+1]-z_scan[idx]))
            def do_nocross(_):
                return jnp.abs(z_scan[idx_last])
            return lax.cond(has_cross, do_cross, do_nocross, None)
        return lax.cond(jnp.max(Tz)<T_MELT, lambda _:0.0, melted, None)

    @jit
    def melt_dims(params, sigma_m):
        params = jnp.asarray(params, dtype=jnp.float64)
        eta        = params[:, 0]
        alpha_mult = params[:, 1]
        z_scale    = params[:, 2]

        alpha   = base_alpha * alpha_mult
        lam     = 12.0*alpha[:,None]*dt_arr[None,:] + sigma_m**2
        inv_lam = 1.0/(lam+eps)
        lam_pow = jnp.power(lam+eps, 1.5)

        dist_y  = dx_sq[None,:,None] + y2[None,None,:]
        term_y  = jnp.exp(-3.0*dist_y*inv_lam[:,:,None]) / lam_pow[:,:,None]
        sum_y   = jnp.sum(term_y, axis=1)

        dist_z  = dx_sq[None,:,None] + z2[None,None,:]*(z_scale[:,None,None]**2)
        term_z  = jnp.exp(-3.0*dist_z*inv_lam[:,:,None]) / lam_pow[:,:,None]
        sum_z   = jnp.sum(term_z, axis=1)

        pref = pref_const * eta
        Ty   = T0 + sum_y*pref[:,None]*dt_step
        Tz   = T0 + sum_z*pref[:,None]*dt_step

        widths_m = vmap(_width)(Ty)
        depths_m = vmap(_depth)(Tz)
        return widths_m*1e6, depths_m*1e6

    return melt_dims
