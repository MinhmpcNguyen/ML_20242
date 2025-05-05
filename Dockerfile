# Base image với Python
FROM python:3.10-slim

# Cài các gói cần thiết (gồm pip và các system libs cần cho Pillow, h5py)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Tạo thư mục làm việc
WORKDIR /app

# Copy mã nguồn vào container
COPY . /app

# Cài Python packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Mở cổng Gradio mặc định
EXPOSE 7860

# Chạy ứng dụng
CMD ["python", "app.py"]