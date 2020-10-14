FROM tensorflow/tensorflow:latest-gpu

RUN mkdir project && chown -R 1000:1000 project
WORKDIR project

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN export PYTHONPATH=$PYTHONPATH:/project/libs

RUN mkdir notebooks
WORKDIR notebooks

RUN mkdir /.local && chown -R 1000:1000 /.local

USER 1000:1000

ENTRYPOINT jupyter notebook --no-browser --ip 0.0.0.0 --port 8888