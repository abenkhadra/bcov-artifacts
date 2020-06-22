FROM ubuntu:xenial
LABEL maintainer="khadra@eit.uni-kl.de"
LABEL version="0.1"
LABEL description="Custom Docker image for the artifact evaluation \
submission of paper #208 to ESEC/FSE 2020"
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
python build-essential cmake git
RUN cd /home/ && git clone https://github.com/abenkhadra/bcov-artifacts
RUN cd /home/bcov-artifacts/ && bash install.sh && bash experiment-01.sh

