FROM continuumio/miniconda3


# https://docs.python.org/3/using/cmdline.html#envvar-PYTHONDONTWRITEBYTECODE
# Prevents Python from writing .pyc files to disk
ENV PYTHONDONTWRITEBYTECODE 1

# ensures that the python output is sent straight to terminal (e.g. your container log)
# without being first buffered and that you can see the output of your application (e.g. django logs)
# in real time. Equivalent to python -u: https://docs.python.org/3/using/cmdline.html#cmdoption-u
ENV PYTHONUNBUFFERED 1
ENV ENVIRONMENT prod
ENV TESTING 0


RUN touch /etc/localtime

RUN apt-get update --fix-missing && DEBIAN_FRONTEND=noninteractive apt upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential git curl wget locales && \
    apt-get -y clean &&  rm -rf /var/lib/apt/lists/*

RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

#ENV PATH="/root/miniconda3/bin:$PATH"
#RUN wget \
#    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
#    && bash Miniconda3-latest-Linux-x86_64.sh -b \
#    && rm -f Miniconda3-latest-Linux-x86_64.sh \
#    && conda config --set show_channel_urls True \
#    && conda config --set path_conflict prevent \
#    && conda config --set notify_outdated_conda false \
#    && conda update -c conda-forge --yes --all \
#    && conda install -c conda-forge --yes conda-build conda-verify coverage coverage-fixpaths \
#    && conda clean -tipy
RUN conda update -c conda-forge --yes --all

WORKDIR '/apps'
RUN git clone https://github.com/gcorso/DiffDock.git

ENV DIFFDOCK_HOME "/apps/DiffDock"
ENV PATH "$DIFFDOCK_HOME:$PATH"
ENV PYTHONPATH "$DIFFDOCK_HOME:$PYTHONPATH"
COPY 'app.py' "$DIFFDOCK_HOME/app.py"



WORKDIR ${DIFFDOCK_HOME}
RUN conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c nvidia -c pytorch -y
RUN conda install pyg -c pyg && \
    conda install rdkit scipy pyyaml networkx biopython biopandas spyrmsd -c conda-forge -y && \
    pip install e3nn gradio

SHELL ["/bin/bash", "-c"]
RUN git clone https://github.com/facebookresearch/esm && cd esm && pip install -e .

RUN mkdir -p ${DIFFDOCK_HOME}/esm/model_weights/.cache/torch/hub/checkpoints && \
    cd ${DIFFDOCK_HOME}/esm/model_weights/.cache/torch/hub/checkpoints && \
    curl -L -O https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt && \
    curl -L -O https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt


WORKDIR "/tmp"
