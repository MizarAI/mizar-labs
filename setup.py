# -*- coding: utf-8 -*-
from setuptools import find_packages
from setuptools import setup

__version__ = "0.1.1"

with open("requirements.txt") as f:
    requirements = f.readlines()

setup(
    name="mizar-labs",
    version=__version__,
    packages=find_packages(),
    include_package_data=True,
    author_email="jack@mizar.ai, cuici@mizar.ai, alex@mizar.ai, cino@mizar.ai,",
    install_requires=requirements,
)
