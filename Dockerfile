FROM tensorflow/tensorflow:latest-gpu

RUN mkdir project
WORKDIR project

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN mkdir notebooks
WORKDIR notebooks

ENTRYPOINT jupyter notebook --no-browser --ip 0.0.0.0 --port 8888 --allow-root