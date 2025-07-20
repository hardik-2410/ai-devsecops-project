# Use slim Python base image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy only the necessary folders
COPY ./app ./app
COPY ./model ./model

# Install required Python packages
RUN pip install fastapi uvicorn scikit-learn pandas joblib

# Expose the port FastAPI will run on
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "asyncio"]

