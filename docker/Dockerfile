FROM mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04

RUN apt-get update && apt-get autoremove -y && env ACCEPT_EULA=Y apt-get upgrade -y

ENV PYTHONPATH "${PYTHONPATH}:${HOME}/take6/"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python-opengl \
    rsync && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /usr/share/man/*

ENV PYTHON_ENV /.virtualenvs/py38

RUN conda install conda-build \
    && conda create -y -p $PYTHON_ENV python=3.8.12 \
    && conda clean -ya

ENV PATH $PYTHON_ENV/bin:$PATH

RUN python -m pip install --no-cache-dir \
    azureml-core==1.35.0 \
    ray[rllib]==2.0.0 \
    statsmodels==0.13.0 \
    openpyxl==3.0.9 \
    tensorflow==2.4.3 \
    gym==0.21.0 \
    trueskill==0.4.5
