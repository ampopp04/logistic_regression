from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

setup(
    name='Logistic Regression',
    version='1.0.0',
    description='A simple neuron implementation using logistic regression and gradient descent.',
    long_description=readme,
    author='Andrew Popp',
    author_email='ampopp04@gmail.com',
    url='https://github.com/ampopp04/logistic_regression',
    packages=find_packages(exclude=('data_sets', 'loop_implementation'))
)