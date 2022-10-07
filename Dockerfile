FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

RUN touch /etc/localtime

RUN apt update && apt upgrade -y && \
    apt install -y --no-install-recommends tini build-essential python3 python3-venv git curl locales

RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8   

RUN mkdir -p /usr/local/apps
WORKDIR /usr/local/apps
RUN git clone https://github.com/gcorso/DiffDock.git
ENV DIFFDOCK_HOME "/usr/local/apps/DiffDock"

WORKDIR ${DIFFDOCK_HOME}
RUN python3 -m venv .venv && \
    . ./.venv/bin/activate

RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116 \
    && python3 -m pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu116.html \
    && python3 -m pip install torch-sparse -f https://data.pyg.org/whl/torch-1.12.1+cu116.html \
    && python3 -m pip install torch-geometric \
    && python3 -m pip install torch-cluster -f https://data.pyg.org/whl/torch-1.12.1+cu116.html \
    && python3 -m pip install torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.1+cu116.html \
    && python3 -m pip install pyyaml scipy networkx biopython rdkit-pypi e3nn spyrmsd pandas biopandas \

RUN git clone https://github.com/facebookresearch/esm && \
    cd ${DIFFDOCK_HOME}/esm && \
    python3 -m pip install -e .

RUN mkdir -p ${DIFFDOCK_HOME}/esm/model_weights/.cache/torch/hub/checkpoints && \
    cd ${DIFFDOCK_HOME}/esm/model_weights/.cache/torch/hub/checkpoints && \
    curl -L -O https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt && \
    curl -L -O https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt

ENV PYTHONPATH "${DIFFDOCK_HOME}"
ENV PYTHONPATH "$PYTHONPATH:${DIFFDOCK_HOME}/esm"

WORKDIR /tmp

ENTRYPOINT ["/tini", "--"]
