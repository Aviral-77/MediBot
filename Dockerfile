FROM python:3.10-slim

WORKDIR /app
COPY . /app

# Upgrade pip BEFORE installing requirements
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python3", "app.py"]