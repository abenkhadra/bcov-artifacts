# Experimental artifacts

This repository contains artifacts accompanying our paper "Efficient Binary-Level Coverage Analysis"
which is accepted to ESEC/FSE'20. It is organized as follows:

  - `sample-binaries`. This folder contains sample binaries patched with our tool.

  - `dataset.tar.gz`. This package contains experimental data in csv format.

  - `figures`. This folder contains a python script used to generate the figures
  of our paper. It assumes the dataset was first extracted to folder `dataset`

  - `install.sh`. This script builds and installs `bcov` together with its dependencies.

  - `experiment-01.sh`. This script patches our sample binaries and shows how coverage
  data can be collected. It assumes that `bcov` was installed using the previous script.

  - `Dockerfile`. A file to build and install the artifacts in a docker image.
