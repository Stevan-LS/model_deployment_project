# Use an official Python image
FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
COPY . .

EXPOSE 8082

CMD python project.py && evidently ui --host 0.0.0.0 --port 8082
