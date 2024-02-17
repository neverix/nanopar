FROM nvcr.io/nvidia/pytorch:24.01-py3
ADD . /content
WORKDIR /content
RUN ./deps.sh