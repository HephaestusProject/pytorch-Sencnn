FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
MAINTAINER bsk0130@gmail.com

RUN apt-get update && \
      apt-get install -y sudo apt-utils make build-essential \
      libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
      libsqlite3-dev wget curl git libffi-dev liblzma-dev locales \
      g++ openjdk-8-jdk openssh-server

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
RUN pyenv install 3.8.5 && \
    pyenv global 3.8.5 $$ \
    pyenv rehash
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

SHELL ["/bin/bash", "-c"]
RUN pip install --no-cache-dir konlpy
RUN bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
RUN pip install --no-cache-dir torch pytorch-lightning mxnet gluonnlp omegaconf pytest pandas scikit-learn

# WORKDIR 설정
WORKDIR /root/workspace

# openssh-server
ARG PASSWD
RUN echo  root:${PASSWD:-hephaestus} | chpasswd
RUN sed -i 's_/usr/lib/openssh/sftp-server_internal-sftp_g' /etc/ssh/sshd_config
RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config
RUN echo "service ssh start" > /root/.bashrc

