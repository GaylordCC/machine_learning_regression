# Use the Python 3.12.1 slim image
FROM python:3.12.1-slim

# Avoid .pyc files and enable unbuffered logs for Uvicorn
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Define the working directory inside the container
WORKDIR /app

# Copy the dependencies file and install them (cached as long as requirements.txt doesn't change)
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files in the current directory to the container
COPY . .

# Run the container as a non-root user
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

# Expose the port the app listens on (see CMD below)
EXPOSE 8080

# Command to start the FastAPI application using Uvicorn
CMD ["uvicorn", "machine_learning.main:app", "--host", "0.0.0.0", "--port", "8080"]