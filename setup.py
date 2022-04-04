import os
import sys
import shutil
from setuptools import setup

if sys.version_info.major != 3:
    raise RuntimeError("TreeAlign requires Python 3")

with open("README.md", "r") as fh:
    long_description = fh.read()    

setup(
    name="treealign",
    version="1.0",
    description=(
        "TreeAlign algorithm for scDNA & scRNA integration"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Hongyu Shi",
    author_email="shih@mskcc.org",
    package_dir={"": "src"},
    packages=["treealign"],
    install_requires=[
        "numpy>=1.12.0",
        "pandas>=0.19.2",
        "scipy>=0.18.1",
        "torch>=1.7.1",
        "pyro-ppl>=1.5.1",
        "biopython>=1.79",
        "simplejson"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.6",
)

# get location of setup.py
setup_dir = os.path.dirname(os.path.realpath(__file__))

