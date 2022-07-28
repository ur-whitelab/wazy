import os
from glob import glob
from setuptools import setup

exec(open("wazy/version.py").read())

setup(
    name="wazy",
    version=__version__,
    description="Pretrained Bayesian Optimization of Sequences",
    author="Ziyue Yang, Andrew White",
    author_email="andrew.white@rochester.edu",
    url="http://thewhitelab.org/Software",
    license="MIT",
    packages=["wazy"],
    install_requires=[
        "jax",
        "dm-haiku",
        "optax",
        "numpy",
        "jax-unirep@git+https://github.com/ElArkk/jax-unirep.git",
    ],
    test_suite="tests",
    zip_safe=True,
)
