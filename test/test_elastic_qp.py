import qpax 

import jax 
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np 


def load_problem_data():
    d = np.load("test/problem_data.npz")
    Q = jnp.array(d["Q"])
    q = jnp.array(d["q"])
    G = jnp.array(d["G"])
    h = jnp.array(d["h"])
    penalty = d["penalty"]
    
    return Q, q, G, h, penalty 

def load_initial_values():
    d = np.load("initial_values.npz")
    x = jnp.array(d["x"])
    t = jnp.array(d["t"])
    s1 = jnp.array(d["s1"])
    s2 = jnp.array(d["s2"])
    z1 = jnp.array(d["z1"])
    z2 = jnp.array(d["z2"])

    
    return x, t, s1, s2, z1, z2

def test_with_qpax():
    
    Q, q, G, h, penalty  = load_problem_data()

    print(dir(qpax))
    
    x, t, s1, s2, z1, z2, converged, pdip_iter = jax.jit(qpax.solve_qp_elastic)(Q,q,G,h, penalty, solver_tol=1e-8)
    
    assert converged == 1
    # solve with enlarged qpax solver 
    nx = len(q)
    ns = len(h)
    zxs = jnp.zeros((nx, ns))
    zxx = 0 * jnp.eye(nx)
    zss = 0 * jnp.eye(ns)
    zsx = jnp.zeros((ns, nx))
    Q2 = jnp.block([
        [Q,     zxs],
        [zsx, zss]
    ])
    q2 = jnp.concatenate((q, -penalty * jnp.ones(ns)))
    
    G2 = jnp.block([
        [zsx, jnp.eye(ns)],
        [G,   jnp.eye(ns)]
    ])
    h2 = jnp.block([jnp.zeros(ns), h])
    
    A2 = jnp.zeros((0, nx + ns))
    b2 = jnp.zeros(0)
    
    x_qpax = qpax.solve_qp_primal(Q2, q2, A2, b2, G2, h2, solver_tol=1e-8)
    
    assert jnp.linalg.norm(x_qpax - jnp.concatenate((x, t))) < 1e-6

test_with_qpax()

def test_diff_with_qpax():
    
    Q, q, G, h, penalty  = load_problem_data()
    
    x, t, s1, s2, z1, z2, converged, pdip_iter = jax.jit(qpax.solve_qp_elastic)(Q,q,G,h, penalty, solver_tol=1e-3)
    
    assert converged == 1 
    
    # solve with enlarged qpax solver 
    nx = len(q)
    ns = len(h)
    zxs = jnp.zeros((nx, ns))
    zxx = 0 * jnp.eye(nx)
    zss = 0 * jnp.eye(ns)
    zsx = jnp.zeros((ns, nx))
    Q2 = jnp.block([
        [Q,     zxs],
        [zsx, zss]
    ])
    q2 = jnp.concatenate((q, -penalty * jnp.ones(ns)))
    
    G2 = jnp.block([
        [zsx, jnp.eye(ns)],
        [G,   jnp.eye(ns)]
    ])
    h2 = jnp.block([jnp.zeros(ns), h])
    
    A2 = jnp.zeros((0, nx + ns))
    b2 = jnp.zeros(0)
    
    x_qpax = qpax.solve_qp_primal(Q2, q2, A2, b2, G2, h2)
    
    assert jnp.linalg.norm(x_qpax - jnp.concatenate((x, t))) < 1e-3
    
    x_des = jnp.array(np.random.randn(len(q)))
    
    def testf_elastic(_Q, _q, _G, _h):
        _x = qpax.solve_qp_elastic_primal(_Q,_q,_G,_h, penalty, solver_tol=1e-3, target_kappa=1e-3)
        return jnp.sum((_x - x_des)**2)
    
    def testf_og(_Q, _q, _G, _h):
        nx = len(_q)
        ns = len(_h)
        zxs = jnp.zeros((nx, ns))
        zxx = 0 * jnp.eye(nx)
        zss = 0 * jnp.eye(ns)
        zsx = jnp.zeros((ns, nx))
        Q2 = jnp.block([
            [_Q,     zxs],
            [zsx, zss]
        ])
        q2 = jnp.concatenate((_q, -penalty * jnp.ones(ns)))

        G2 = jnp.block([
            [zsx, jnp.eye(ns)],
            [_G,   jnp.eye(ns)]
        ])
        h2 = jnp.block([jnp.zeros(ns), _h])

        A2 = jnp.zeros((0, nx + ns))
        b2 = jnp.zeros(0)

        _x_qpax = qpax.solve_qp_primal(Q2, q2, A2, b2, G2, h2, solver_tol=1e-3, target_kappa=1e-3)
        return jnp.sum((_x_qpax[:nx] - x_des)**2)
    
    grads1 = jax.jit(jax.grad(testf_elastic, (0, 1, 2, 3)))(Q, q, G, h)
    grads2 = jax.jit(jax.grad(testf_og, (0, 1, 2, 3)))(Q, q, G, h)

    
    print("norm of the gradients: ")
    print(jnp.linalg.norm(grads1[0]))
    print(jnp.linalg.norm(grads1[1]))
    print(jnp.linalg.norm(grads1[2]))
    print(jnp.linalg.norm(grads1[3]))
    
    print("norm of the errors:")
    print(jnp.linalg.norm(grads1[0] - grads2[0]))
    print(jnp.linalg.norm(grads1[1] - grads2[1]))
    print(jnp.linalg.norm(grads1[2] - grads2[2]))
    print(jnp.linalg.norm(grads1[3] - grads2[3]))

    assert jnp.linalg.norm(grads1[0] - grads2[0]) < 1e-3
    assert jnp.linalg.norm(grads1[1] - grads2[1]) < 1e-3
    assert jnp.linalg.norm(grads1[2] - grads2[2]) < 1e-3
    assert jnp.linalg.norm(grads1[3] - grads2[3]) < 1e-3


test_diff_with_qpax()