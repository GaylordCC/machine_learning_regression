# Use the Python 3.12.1 slim image
FROM python:3.12.1-slim

# Define the working directory inside the container
WORKDIR /app

# Copy the dependencies file and install them
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy all application files in the current directory to the container
COPY . .

# Expose port 8000, which is the default port for Uvicorn
EXPOSE 8000

# Command to start the FastAPI application using Uvicorn
CMD ["uvicorn", "machine_learning.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

