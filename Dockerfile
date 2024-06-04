FROM python:3.10

COPY requirements.txt /tmp/
RUN apt-get update && apt-get install libgl1-mesa-glx --yes
RUN apt-get install poppler-utils
RUN pip install torch
RUN pip install -U 'git+https://github.com/nikhilweee/iopath'
RUN pip install -r /tmp/requirements.txt

WORKDIR /app
COPY . .
RUN curl -SL https://minio.lab.sspcloud.fr/ayoub1/model_final.pth -o model_final.pth

CMD ["streamlit", "run", "main.py","--server.port", "3838"]
