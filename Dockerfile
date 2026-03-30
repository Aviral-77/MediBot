FROM python:3.10-slim

# Set environment variables to keep Python clean
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 1. Copy ONLY requirements.txt first
COPY requirements.txt .

# 2. Upgrade pip and install dependencies (This layer is now isolated)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 3. NOW copy the rest of your application code
COPY . .

# Expose the port your app runs on
EXPOSE 8080

CMD ["python3", "app.py"]