
FROM python:3.10.16-slim


RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app


COPY . /app


RUN pip install --upgrade pip
RUN pip install -r requirements.txt



EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["sh", "-c", "python -u hugging_face_download.py && python -u app.py"]