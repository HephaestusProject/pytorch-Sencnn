FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
MAINTAINER bsk0130@gmail.com

RUN apt-get update && \
      apt-get install -y sudo apt-utils make build-essential \
      libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
      libsqlite3-dev wget curl git libffi-dev liblzma-dev locales \
      g++ openjdk-8-jdk

RUN locale-gen en_US.UTF-8
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8

USER root
WORKDIR /root

# pyenv 설치/ 설정
ENV HOME /root
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN git clone https://github.com/pyenv/pyenv.git .pyenv

# python 설치
RUN pyenv install 3.8.5
RUN pyenv global 3.8.5
RUN pyenv rehash
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install --upgrade wheel

SHELL ["/bin/bash", "-c"]
RUN pip install konlpy
RUN bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
RUN pip install torch
RUN pip install pytorch-lightning
RUN pip install mxnet
RUN pip install gluonnlp
RUN pip install omegaconf
RUN pip install pytest
RUN pip install pandas

# WORKDIR 설정
WORKDIR /root/workspace

