# Dockerfile for prodigy, just place your linux-wheel (prodigy-0.1.0-cp36-cp36m-linux_x86_64.whl) in same directoty as
# this dockerfile and:
# > docker build . -t slovak-ner/prodigy:version
# > docker run -it -p 8080:8080 -v ${PWD}:/data slovak-ner/prodigy:version bash

FROM python:3.6
RUN mkdir /prodigy
WORKDIR /prodigy
COPY ./prodigy-1.10.8-cp36.cp37.cp38-cp36m.cp37m.cp38-linux_x86_64.whl /prodigy
RUN pip install prodigy-1.10.8-cp36.cp37.cp38-cp36m.cp37m.cp38-linux_x86_64.whl
RUN mkdir /data
ENV PRODIGY_HOME /data
WORKDIR /data
ENTRYPOINT ["tail", "-f", "/dev/null"]
