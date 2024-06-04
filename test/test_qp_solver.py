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

from misc_test_utils import generate_random_qp
from misc_test_utils import check_kkt_conditions


def test_qp_solver():

	np.random.seed(1)

	# test 1000 normal QP's 
	nx = 15
	ns = 10
	nz = ns 
	ny = 3

	# jit compile first 
	jit_solve_qp = jax.jit(qpax.solve_qp)

	for i in range(1000):
		Q, q, A, b, G, h, x_true, s_true, z_true, y_true = generate_random_qp(nx, ns, ny)
		x,s,z,y,iters = jit_solve_qp(Q,q,A,b,G,h)

		duality_gap = s.dot(z)/len(s)
		ineq_res = jnp.linalg.norm(G @ x + s - h, ord = jnp.inf)

		assert iters <= 6 # this is much stricter than the continuation criteria
		assert duality_gap <= 1e-4 
		assert ineq_res <= 1e-4

		check_kkt_conditions(Q,q,A,b,G,h,x,s,z,y)
		

	# test 1000 inequality-only QP's 
	nx = 15
	ns = 10
	nz = ns 
	ny = 0

	for i in range(1000):
		Q, q, A, b, G, h, x_true, s_true, z_true, y_true = generate_random_qp(nx, ns, ny)
		x,s,z,y,iters = jit_solve_qp(Q,q,A,b,G,h)

		duality_gap = s.dot(z)/len(s)
		ineq_res = jnp.linalg.norm(G @ x + s - h, ord = jnp.inf)

		assert iters <= 10 # this is much stricter than the continuation criteria
		assert duality_gap <= 1e-4 
		assert ineq_res <= 1e-4

		check_kkt_conditions(Q,q,A,b,G,h,x,s,z,y)
