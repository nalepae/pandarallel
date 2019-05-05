from setuptools import setup, find_packages

install_requires = [
    'pandas',
    'pyarrow >= 0.12.1',
    'tqdm >= 4.31.1',
]

setup(
    name='pandarallel',
    version='1.1.1',
    python_requires='>=3.5',
    packages=find_packages(),
    author='Manu NALEPA',
    author_email='nalepae@gmail.com',
    description='An easy to use library to speed up computation (by parallelizing on multi CPUs) with pandas.',
    long_description='See [here](https://github.com/nalepae/pandarallel/tree/v1.0.0) for complete user guide.',
    url='https://github.com/nalepae/pandarallel',
    install_requires=install_requires,
    license='BSD',
)
