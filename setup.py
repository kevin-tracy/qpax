import os
from setuptools import find_packages
from setuptools import setup


root_path = os.path.dirname(__file__)
version_path = os.path.join(root_path, 'qpax', 'version.py')

_dct = {}
with open(version_path) as f:
  exec(f.read(), _dct)
__version__ = _dct['__version__']

req_path = os.path.join(root_path, 'requirements.txt')
install_requires = []
if os.path.exists(req_path):
  with open(req_path) as fp:
    install_requires = [line.strip() for line in fp]

setup(
    name='qpax',
    version=__version__,
    description='Differentiable QP solver in JAX.',
    author='Kevin Tracy',
    author_email='ktracy@cmu.edu',
    url='https://github.com/kevin-tracy/qpax',
    license='MIT',
    packages=find_packages(),
    package_data={},
    install_requires=install_requires,
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    keywords='optimization, automatic differentiation, jax',
    requires_python='>=3.7',
)