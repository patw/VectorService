FROM ubuntu:22.10
ENV CONTAINER_SHELL=bash
ENV CONTAINER=

ARG DEBIAN_FRONTEND=noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# basic app installs
# two steps because i kept getting cache errors without it
RUN apt-get clean && \
    apt-get update
RUN apt-get install -y \
        python3.11  \
        wget \ 
        git \
        rustc \
        cargo \
        openssl \
        libssl-dev \
        pkg-config \
        build-essential

# links
RUN ln -s /usr/bin/python3.11 /usr/bin/python3 -f
RUN ln -s /usr/bin/python3.11 /usr/bin/python -f

# install pip
RUN wget -O get-pip.py https://bootstrap.pypa.io/get-pip.py
RUN python3 get-pip.py

RUN mkdir /opt/vectorservice
COPY ./main.py /opt/vectorservice/main.py
COPY ./minvec.py /opt/vectorservice/minvec.py
COPY ./requirements.txt /opt/vectorservice/requirements.txt

# install pip required packages
RUN python3 -m pip install -r /opt/vectorservice/requirements.txt
RUN python3 -m spacy download en_core_web_sm
RUN python -m spacy download en_core_web_md
RUN python -m spacy download en_core_web_lg

RUN git clone https://github.com/naver/splade.git /opt/splade
RUN cd /opt/splade && python setup.py install

COPY ./preloadpackages.py /opt/vectorservice/preloadpackages.py

RUN cd /opt/vectorservice
WORKDIR /opt/vectorservice
RUN python3 preloadpackages.py
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
#CMD ["/bin/bash"]

EXPOSE 80
