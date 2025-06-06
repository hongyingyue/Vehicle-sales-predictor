FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AIRFLOW_HOME=/opt/airflow

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements files
COPY requirements.txt .
COPY airflow/requirements.txt ./airflow-requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r airflow-requirements.txt \
    && pip install --no-cache-dir apache-airflow==2.7.1 \
    && pip install --no-cache-dir uvicorn

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p ${AIRFLOW_HOME}/dags \
    && mkdir -p ${AIRFLOW_HOME}/logs \
    && mkdir -p ${AIRFLOW_HOME}/plugins \
    && mkdir -p /app/models \
    && mkdir -p /app/data

# Set permissions
RUN chmod -R 755 ${AIRFLOW_HOME}

# Expose ports
EXPOSE 8080 8000

# Set default command
CMD ["bash"]
