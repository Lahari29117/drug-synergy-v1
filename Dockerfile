# Use the same Python major/minor version as your local environment.
# If "python --version" says 3.11, change 3.12 below to 3.11.
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install the exact packages saved in requirements.txt.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy only the files needed while the deployed app is running.
COPY app ./app
COPY models ./models

EXPOSE 8000

# Railway injects PORT in production; 8000 keeps local Docker testing simple.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]