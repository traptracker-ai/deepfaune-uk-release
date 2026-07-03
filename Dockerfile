FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        git libgl1 libglib2.0-0 ffmpeg nano && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace
EXPOSE 8888
CMD ["bash", "-lc", "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.root_dir=/workspace --ServerApp.token='' --ServerApp.password=''"]
