from setuptools import setup, find_packages

version = '0.0.6'

setup(

    #author info
    author='Colin Dickson',
    author_email='colin.dickson@gameloft.com',

    #license
    license='GPL',

    #package info
    name='koala',
    version=version,
    packages=find_packages(),
    description="",

    #dependencies
    install_requires=['numpy >= 1.7.0',
                      'pandas >= 0.12.0',
                      'copper >= 0.0.3']
)
