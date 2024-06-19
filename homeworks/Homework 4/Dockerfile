FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

COPY [ "env.yaml", "script_docker.py", "./"  ]

RUN apt-get update \
    && apt-get install wget -y

RUN rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh
RUN conda --version

RUN conda env create -f env.yaml

RUN echo "conda activate model_docker" > ~/.bashrc
RUN echo "python script_docker.py" > ~/.bashrc
ENV PATH /opt/conda/envs/model_docker/bin:$PATH

EXPOSE 5555

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:5555", "predict:app" ]
