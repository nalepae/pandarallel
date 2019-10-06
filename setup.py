from setuptools import setup, find_packages

install_requires = [
    'pandas',
    'pyarrow >= 0.12.1',
    'pathos >= 0.2.4'
]

setup(
    name='pandarallel',
    version='1.3.3',
    python_requires='>=3.5',
    packages=find_packages(),
    author='Manu NALEPA',
    author_email='nalepae@gmail.com',
    description='An easy to use library to speed up computation (by parallelizing on multi CPUs) with pandas.',
    long_description='See https://github.com/nalepae/pandarallel/tree/v1.3.3 for complete user guide.',
    url='https://github.com/nalepae/pandarallel',
    install_requires=install_requires,
    license='BSD',
)
