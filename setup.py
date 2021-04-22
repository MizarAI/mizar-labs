from setuptools import find_packages
from setuptools import setup

__version__ = "0.1.11"

with open("requirements.txt") as f:
    requirements = f.readlines()

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mizar-labs",
    version=__version__,
    packages=find_packages(),
    include_package_data=True,
    author="MizarAI",
    author_email="info@mizar.ai",
    install_requires=requirements,
    description="Package for building financial machine learning models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MizarAI/mizar-labs",
    project_urls={
        "Bug Tracker": "https://github.com/MizarAI/mizar-labs/issues",
    },
    python_requires=">=3.8",
)
