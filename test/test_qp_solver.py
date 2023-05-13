import qpax 

import jax 
import jax.numpy as jnp
# import jax.scipy as jsp 
# from jax import grad 
# from jax import jacobian 
# from jax import jit 
# from jax import vmap 
import numpy as np 
# from jax import custom_vjp



def gen_qp(nx, ns, ny):

	nz = ns 

	x = jnp.array(np.random.randn(nx))
	Q = jnp.array(np.random.randn(nx,nx))
	Q = Q.T @ Q 
	G = jnp.array(np.random.randn(ns,nx))
	A = jnp.array(np.random.randn(ny,nx))

	s = np.abs(np.random.randn(ns))
	z = np.abs(np.random.randn(ns))
	for i in range(ns):
		if (np.random.rand() < .5):
			s[i] = 0 
		else:
			z[i] = 0 

	s = jnp.array(s)
	z = jnp.array(z)

	h = G @ x + s 

	b = A @ x 
	y = jnp.array(np.random.randn(ny))
	q = -Q @ x - G.T @ z - A.T @ y 

	return Q, q, A, b, G, h, x, s, z, y 


def test_qp_solver():
	assert 4 == 4


	np.random.seed(1)

	nx = 15
	ns = 10
	nz = ns 
	ny = 3 

	for i in range(1000):
		Q, q, A, b, G, h, x_true, s_true, z_true, y_true = gen_qp(nx, ns, ny)
		x,s,z,y,iters = qpax.pdip.solve_qp(Q,q,A,b,G,h)

		duality_gap = s.dot(z)/len(s)
		ineq_res = jnp.linalg.norm(G @ x + s - h, ord = jnp.inf)

		assert iters <= 6 # this is much stricter than the continuation criteria
		assert duality_gap <= 1e-4 
		assert ineq_res <= 1e-4
