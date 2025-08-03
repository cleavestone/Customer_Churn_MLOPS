FROM python:3.10-slim

WORKDIR /app

# Install git and curl (good practice for pip installs + networking)
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


COPY . .

EXPOSE 5000

CMD ["python", "App.py"]
