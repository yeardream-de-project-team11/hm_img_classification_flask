FROM python:3.10.13-slim

WORKDIR /usr/src/app

COPY . .

RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

WORKDIR ./myapp

ENV AWS_ACCESS_KEY_ID=access_key
ENV AWS_SECRET_ACCESS_KEY=secret_key
ENV MODEL_RUN_ID=runid
ENV MLFLOW_URI=http://mlflow:5000


EXPOSE 5000

CMD ["python", "main.py"]