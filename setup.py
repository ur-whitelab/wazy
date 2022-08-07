import os
from glob import glob
from setuptools import setup

exec(open("wazy/version.py").read())

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


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
        "jax-unirep",
    ],
    test_suite="tests",
    zip_safe=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
