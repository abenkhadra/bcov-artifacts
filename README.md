# Experimental artifacts

This repository contains the artifacts accompanying the tool [bcov](https://github.com/abenkhadra/bcov) 
which is presented in our paper "Efficient Binary-Level Coverage Analysis"  (accepted to ESEC/FSE'20). 
The artifacts are organized as follows:

  - `sample-binaries`. This folder contains sample binaries patched with our tool.

  - `dataset.tar.gz`. This package contains experimental data in csv format.

  - `figures`. This folder contains a python script used to generate the figures
  of our paper. It assumes the dataset was first extracted to folder `dataset`

  - `install.sh`. This script builds and installs **bcov** together with its dependencies.

  - `experiment-01.sh`. This script patches the sample binaries and shows how coverage
  data can be collected. It assumes that **bcov** was installed using `install.sh`.

  - `Dockerfile`. A file to build and install the artifacts in a docker image.

The docker image can be built by running,

```bash
sudo docker build  --tag bcov-image:0.1 https://raw.githubusercontent.com/abenkhadra/bcov-artifacts/master/Dockerfile
```

Then, the installed artifacts can be examined,

```bash
sudo docker run -w /home/bcov-artifacts -it bcov-image:0.1
```
