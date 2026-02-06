# 1. Use Python 3.11 (Slim version for faster builds)
FROM python:3.11-slim

# 2. Set the working directory
WORKDIR /app

# 3. Install only necessary system tools (build-essential is needed for FAISS)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy your project files
COPY . .

# 5. Install Python dependencies
# Using --no-cache-dir saves space and prevents some build errors
RUN pip install --no-cache-dir -r requirements.txt

# 6. Expose the port Hugging Face expects
EXPOSE 7860

# 7. Run Streamlit
# We use the flags to ensure it runs correctly in a container environment
CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0", "--server.headless", "true"]