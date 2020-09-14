import os
from glob import glob
from setuptools import setup

exec(open('alpdesign/version.py').read())

setup(name='alpdesign',
      version=__version__,
      scripts=glob(os.path.join('scripts', '*')),
      description='Active Learning Peptide',
      author='Ziyue Yang, Rainier Barrett, Andrew White',
      author_email='andrew.white@rochester.edu',
      url='http://thewhitelab.org/Software',
      license='MIT',
      packages=['alpdesign'],
      install_requires=[
          'tensorflow >= 2.3',
          'numpy'],
      test_suite='tests',
      zip_safe=True
      )
