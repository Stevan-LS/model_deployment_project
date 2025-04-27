# Use an official Python image
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
COPY . .

EXPOSE 8081

CMD ["streamlit", "run", "api.py", "--server.port=8081", "--server.address=0.0.0.0"]