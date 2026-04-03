# Dockerfile

# Start with a Python box
FROM python:3.10-slim

# Set the working folder inside the box
WORKDIR /app

# Copy the shopping list and install everything
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all our app code into the box
COPY app/ .

# Train the model when building the box
RUN python model.py

# Open door number 5000
EXPOSE 5000

# When the box starts, run our API
CMD ["python", "predict.py"]