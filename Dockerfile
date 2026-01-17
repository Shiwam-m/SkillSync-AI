# Dockerfile for deploying the application on Hugging Face Spaces.
# This file sets up a Python 3.11 environment, installs dependencies,
# and runs the Streamlit application on port 7860.



# 1. Use Python 3.11
FROM python:3.11

# 2. Create the working directory INSIDE the container
WORKDIR /app

# 3. Copy all files from your main directory into the container's /app folder
COPY . .

# 4. Install requirements
RUN pip install --no-cache-dir -r requirements.txt

# 5. Tell Hugging Face to use port 7860
EXPOSE 7860

# 6. Run the app (pointing to app.py which is now in the current WORKDIR)
CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"]
