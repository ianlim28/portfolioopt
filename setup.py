from setuptools import find_packages, setup

setup(name='portfolioopt',
      version='0.1',
      description='Functions for optimizing a portfolio',
      url='https://github.com/ianlim28/portfolioopt',
      author='ianlim',
      author_email='ianlim28@hotmail.com',
      license='MIT',
      packages=find_packages(),
      package_dir={"": "src"},
      zip_safe=False)