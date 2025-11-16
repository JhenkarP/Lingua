# ✅ Use latest PyTorch base image with CUDA
FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# ✅ Install ffmpeg system dependency + Python packages
RUN apt-get update && apt-get install -y ffmpeg && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --timeout 600 && \
    rm -rf /var/lib/apt/lists/*

# Copy your code
COPY . .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


#python -m uvicorn main:app --reload --port 8000 --ws websockets
