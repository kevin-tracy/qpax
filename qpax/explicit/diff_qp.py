import jax
import jax.numpy as jnp

from qpax.explicit.pdip import factorize_kkt, solve_kkt_rhs, solve_qp
from qpax.explicit.pdip_relaxed import relax_qp


def optnet_derivatives(dz, dlam_tilde, dnu, z, lam, nu):
    dl_dQ = 0.5 * (jnp.outer(dz, z) + jnp.outer(z, dz))
    dl_dA = jnp.outer(dnu, z) + jnp.outer(nu, dz)
    dl_dG = jnp.outer(dlam_tilde, z) + jnp.outer(lam, dz)

    dl_dq = dz
    dl_db = -dnu
    dl_dh = -dlam_tilde

    return dl_dQ, dl_dq, dl_dA, dl_db, dl_dG, dl_dh


def diff_qp(Q, q, A, b, G, h, z, s, lam, nu, dl_dz):
    ns = len(h)
    nnu = len(b)
    cotangent_dtype = dl_dz.dtype

    P_inv_vec, L_H, L_F = factorize_kkt(Q, G, A, s, lam)

    dz, ds, dlam_tilde, dnu = solve_kkt_rhs(
        G,
        A,
        s,
        lam,
        P_inv_vec,
        L_H,
        L_F,
        -dl_dz,
        jnp.zeros(ns, dtype=cotangent_dtype),
        jnp.zeros(ns, dtype=cotangent_dtype),
        jnp.zeros(nnu, dtype=cotangent_dtype),
    )

    return optnet_derivatives(dz, dlam_tilde, dnu, z, lam, nu)


@jax.custom_vjp
def solve_qp_primal(Q, q, A, b, G, h, solver_tol=1e-5, target_kappa=1e-3, max_iter=30):

    x, s, z, y, _, _ = solve_qp(
        Q, q, A, b, G, h, solver_tol=solver_tol, max_iter=max_iter
    )
    return x


"""
these two functions are only called when we diff solve_qp_x
"""


def solve_qp_primal_forward(
    Q, q, A, b, G, h, solver_tol=1e-5, target_kappa=1e-3, max_iter=30
):

    x, s, z, y, _, _ = solve_qp(
        Q, q, A, b, G, h, solver_tol=solver_tol, max_iter=max_iter
    )
    xr, sr, zr, yr, _, _ = relax_qp(
        Q,
        q,
        A,
        b,
        G,
        h,
        x,
        s,
        z,
        y,
        solver_tol=solver_tol,
        target_kappa=target_kappa,
        max_iter=max_iter,
    )
    res = (Q, q, A, b, G, h, xr, sr, zr, yr)
    return x, res


def solve_qp_primal_backward(res, input_grad):

    Q, q, A, b, G, h, xr, sr, zr, yr = res

    return (
        *diff_qp(Q, q, A, b, G, h, xr, sr, zr, yr, input_grad),
        None,
        None,
        None,
    )


solve_qp_primal.defvjp(solve_qp_primal_forward, solve_qp_primal_backward)
