FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04

ENV TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update --fix-missing && apt-get install -y fontconfig --fix-missing
RUN apt-get install -y libopenmpi-dev
RUN apt-get install -y git python3.9 python3.9-dev python3.9-venv
RUN python3.9 -m venv /venv
ENV PATH=/venv/bin:$PATH
RUN pip install mpi4py

ARG SRCDIR

RUN mkdir /LMFlow/
WORKDIR /LMFlow/

COPY $SRCDIR/  /LMFlow/

RUN pip install wheel
RUN pip install -e .
