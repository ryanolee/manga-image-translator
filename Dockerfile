FROM pytorch/pytorch:latest

ARG RELEASE_VERSION=beta-0.3
ARG ASSET_BASE_URL=https://github.com/zyddnys/manga-image-translator/releases/download

WORKDIR /app

# Assume root to install required dependencies
RUN apt-get update && \
    apt-get install -y git g++ ffmpeg libsm6 libxext6 && \
    pip install git+https://github.com/lucasb-eyer/pydensecrf.git &&\
    apt-get remove -y git g++

# Install pip dependencies

COPY requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt

# Copy remaing dependencies
ADD ${ASSET_BASE_URL}/${RELEASE_VERSION}/ocr.ckpt \
     ${ASSET_BASE_URL}/${RELEASE_VERSION}/ocr-ctc.ckpt \
     ${ASSET_BASE_URL}/${RELEASE_VERSION}/detect.ckpt \
     ${ASSET_BASE_URL}/${RELEASE_VERSION}/comictextdetector.pt \
     ${ASSET_BASE_URL}/${RELEASE_VERSION}/comictextdetector.pt.onnx \
     ${ASSET_BASE_URL}/${RELEASE_VERSION}/inpainting_lama_mpe.ckpt \
     /app/

# Copy app
COPY . /app

ENTRYPOINT ["python", "-u", "/app/translate_demo.py"]