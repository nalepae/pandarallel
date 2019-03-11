from setuptools import setup, find_packages

import pandarallel

setup(
    name='pandarallel',
    version=pandarallel.__version__,
    packages=find_packages(),
    author='Manu NALEPA',
    author_email='nalepae@gmail.com',
    description='An easy to use library to speed up computation (by parallelizing on multi CPUs) with pandas.',
    long_description=open('README.md').read(),
    url='https://github.com/nalepae/pandarallel',
)