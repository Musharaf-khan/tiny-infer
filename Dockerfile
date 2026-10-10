# GPU image: docker build -t tiny-infer . && docker run --gpus all -p 8000:8000 tiny-infer
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# compile the CUDA kernels at build time so the first request doesn't pay for it
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9"
ENV TINY_INFER_ARCHS=75,80,86,89
RUN python -c "from tiny_infer.kernels import build_extension; build_extension()" || echo "kernel prebuild skipped (no GPU at build time); will JIT on first start"
EXPOSE 8000
CMD ["python", "-m", "tiny_infer.server", "--custom-kernels", "--port", "8000"]
