FROM python:3.10.13-slim

WORKDIR /app

COPY . .

RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt


EXPOSE 5000

CMD ["python", "main.py"]