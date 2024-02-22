FROM nvcr.io/nvidia/pytorch:24.01-py3
RUN mkdir -p /content
WORKDIR /content
COPY ./deps.sh .

RUN ./deps.sh
# only for checking that logprobs match
RUN pip install git+https://github.com/facebookresearch/llama

COPY *.py *.sh ./
COPY models/ ./models/
COPY hh-rlhf/ ./hh-rlhf/
