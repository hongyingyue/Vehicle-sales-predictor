FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy entire project to /app
COPY . .

# Set default working directory to /app/examples
WORKDIR /app/examples

# Run training and then inference
CMD ["sh", "-c", "python run_train.py && python app.py"]
