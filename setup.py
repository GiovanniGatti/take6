#!/usr/bin/env python
from setuptools import setup

try:
    import tensorflow
except ImportError as e:
    raise ModuleNotFoundError('Expected tensorflow>=2.2.0 to be provided')

setup(name='take6',
      version='0.1.0-SNAPSHOT',
      author='Giovanni Gatti Pinheiro',
      author_email='giovanni.gattipinheiro@amadeus.com',
      setup_requires=['wheel'],
      package_dir={'': 'src'},
      packages=['take6'],
      package_data={'take6': ['py.typed']},
      install_requires=['azureml-core==1.35.0', 'azureml-train-core==1.35.0', 'ray[rllib]==1.7.0', 'gym==0.21.0'],
      python_requires='>=3.7')
