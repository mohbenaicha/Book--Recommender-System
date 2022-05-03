#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from setuptools import find_packages, setup


other = {}
ROOT_DIR = Path(__file__).resolve().parent
REQUIREMENTS_DIR = ROOT_DIR / 'requirements'
PACKAGE_DIR = ROOT_DIR / 'recommender_model'
with open(PACKAGE_DIR / "VERSION") as f:
    _version = f.read().strip()
    other["__version__"] = _version


def list_reqs(filename="requirements.txt"):
    ''' Install requirements for setup script '''
    with open(REQUIREMENTS_DIR / filename) as f:
        return f.read().splitlines()


setup(
    name='book_recommender',
    version=other["__version__"],
    description="A DNN book recommender based on collaborative filtering",
    long_description="""A book recommender trained on ~ 1 million raw data records
    and utilizes DNN embedding layers that enherit from the tesnorflow recommenders
    Model.""",
    long_description_content_type="text/markdown",
    author="Mohamed Benaicha",
    author_email="mohamed.benaicha@hotmail.com",
    python_requires=">=3.9.5",
    packages=find_packages(exclude=("tests",)),
    package_data={"recommender_model": ["VERSION"]},
    install_requires=list_reqs(), # in place of setup_requires as per PEP 517
    extras_require={},
    include_package_data=True,
    license='Apache License 2.0',
    license_files=("LICENSE.txt",), 
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Topic :: Internet :: WWW/HTTP :: WSGI :: Application"
    ])