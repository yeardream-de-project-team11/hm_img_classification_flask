FROM python:3.10.13-slim

WORKDIR /usr/src/app

COPY . .

RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR ./myapp

EXPOSE 5000

CMD ["python", "main.py"]