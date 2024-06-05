import jax 
import jax.numpy as jnp
import numpy as np 


def generate_random_qp(nx, ns, ny):

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


def check_kkt_conditions(Q,q,A,b,G,h,x,s,z,y, solver_tol=1e-3):
	r1 = Q @ x + q + A.T @ y + G.T @ z 
	r2 = s * z 
	r3 = G @ x + s - h 
	r4 = A @ x - b

	assert jnp.linalg.norm(r1, ord = jnp.inf) <= solver_tol
	assert jnp.linalg.norm(r2, ord = jnp.inf) <= solver_tol
	assert jnp.linalg.norm(r3, ord = jnp.inf) <= solver_tol

	if (len(b) > 0):
		assert jnp.linalg.norm(r4, ord = jnp.inf) <= solver_tol


def vectorize_mat(mat):
	r,c = mat.shape 
	return jnp.reshape(mat, (r*c,))

def materize_vec(vec, dims):
	return jnp.reshape(vec, dims)


def finite_difference(func, x):
	y = func(x)
	dx = 1e-3
	# finite diff of a vector 
	if len(x.shape) == 1: 
	
		g = jnp.zeros(len(x))

		

		for i in range(len(x)):

			x2 = x.at[i].set(x[i] + dx)
			y2 = func(x2)

			g = g.at[i].set((y2 - y)/dx)

		return g 
	else:

		nr,nc = x.shape
		nel = nr * nc 
		g = jnp.zeros(nel)

		for i in range(nel):

			x2 = vectorize_mat(x)
			x2 = x2.at[i].set(x2[i] + dx)

			y2 = func(materize_vec(x2, (nr,nc)))

			g = g.at[i].set((y2 - y)/dx)


		return materize_vec(g, (nr, nc))