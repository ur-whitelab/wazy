import os
from glob import glob
from setuptools import setup

exec(open("alpdesign/version.py").read())

setup(
    name="alpdesign",
    version=__version__,
    description="Active Learning Peptide",
    author="Ziyue Yang, Andrew White",
    author_email="andrew.white@rochester.edu",
    url="http://thewhitelab.org/Software",
    license="MIT",
    packages=["alpdesign"],
    install_requires=[
        "jax",
        "dm-haiku==0.0.6",
        "optax",
        "numpy",
        "jax-unirep@git+https://github.com/ElArkk/jax-unirep.git",
    ],
    test_suite="tests",
    zip_safe=True,
)
