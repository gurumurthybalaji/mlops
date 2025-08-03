# Use an official Python base image
FROM python:3.11

# Set working directory in the container
WORKDIR /app

# Copy everything from the current folder to the container
COPY . .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the FastAPI port
EXPOSE 8000

# Run the API using uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
