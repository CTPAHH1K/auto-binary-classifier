import setuptools
from setuptools import setup

setup(name='automl_pn',
      version='0.1.3',

      description='Automated binary classification',
      packages=['automl_pn', 'automl_pn.utils'],
      author_email='maximponomarev92@gmail.com',
      zip_safe=False,
      python_requires='>= 3.7',
      install_requires=[
            "lightgbm",
            "catboost",
            "pandas",
            "numpy",
            "scikit-learn",
      ],
      )
