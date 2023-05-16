# qpax
Differentiable QP solver in [JAX](https://github.com/google/jax).

This package can be used for solving convex quadratic programs of the following form:

$$
\begin{align*}
\underset{x}{\text{minimize}} & \quad \frac{1}{2}x^TQx + q^Tx \\
\text{subject to} & \quad  Ax = b, \\
                  & \quad  Gx \leq h
\end{align*}
$$

where $Q \succeq 0$. This solver can be combined with JAX's `jit` and `vmap` functionality, as well as differentiated with reverse-mode `grad`. 

The QP is solved with a primal-dual interior point algorithmn detailed in [cvxgen](https://stanford.edu/~boyd/papers/pdf/code_gen_impl.pdf), with the solution to the linear systems computed with reduction techniques from [cvxopt](http://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf). At an approximate primal-dual solution, the the primal variable $x$ is differentiated with respect to the problem parameters using the implicit function theorem as shown in [optnet](https://arxiv.org/abs/1703.00443), and their pytorch qp solver [qpth](https://github.com/locuslab/qpth).

## Installation

To install directly from github using `pip`:

```bash
$ pip install git+https://github.com/kevin-tracy/qpax
```

Alternatively, to install from source:

```bash
$ python setup.py install
```

## Usage

We can solve QP's with qpax in a way that plays nice with JAX's `jit` and `vmap`:
```python 
ipmort qpax

# solve QP (this can be combined with jit or vmap)
x, s, z, y, iters = qpax.solve_qp(Q, q, A, b, G, h)
```

Alternatively, if we are only looking to use the primal variable `x`, we can use `solve_qp_x` and enable automatic differenation:

```python
import jax 
import jax.numpy as jnp 
import qpax 

def loss(Q, q, A, b, G, h):
    x = qpax.solve_qp_x(Q, q, A, b, G, h) 
    x_bar = jnp.ones(len(q))
    return jnp.dot(x - x_bar, x - x_bar)
  
# gradient of loss function   
loss_grad = jax.grad(loss, argnums = (0, 1, 2, 3, 4, 5))

# compatible with jit 
loss_grad_jit = jax.jit(loss_grad)

# calculate derivatives 
derivs = loss_grad_jit(Q, q, A, b, G, h)
dl_dQ, dl_dq, dl_dA, dl_db, dl_dG, dl_dh = derivs 
```
