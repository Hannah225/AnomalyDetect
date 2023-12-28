# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

# with open('LICENSE') as f:
#     license = f.read()

setup(
    name='AnomalyDetect',
    version='0.1.0',
    description='Basic Anomaly Detection in Latent Space',
    long_description=readme,
    author='Hannah Blaurock',
    author_email='blaurock.hannah@gmail.com',
    url='https://github.com/Hannah225/AnomalyDetect',
    license=license,
    packages=find_packages(
        #exclude=('tests', 'docs')
        )
)