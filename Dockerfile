# blockyOS in a container.
#   NVIDIA GPU:  docker compose up -d            (needs the NVIDIA Container Toolkit; see docs/SETUP.md)
#   CPU only:    docker build --build-arg BASE=ubuntu:24.04 -t blockyos:cpu . && docker run -d --network host blockyos:cpu
# Then open http://localhost:8000 on that machine. Models and data live in volumes at /opt/blocky/models and /opt/blocky/data.
# The container uses the host's network so Blocky stays reachable from this machine only (its access rules rely on that).
ARG BASE=nvidia/cuda:12.9.1-devel-ubuntu24.04

# ---- The web UI ----
FROM node:22-bookworm-slim AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---- Blocky ----
FROM ${BASE}
# GPU generations to compile for: 61 = GTX 10xx, 75 = RTX 20xx, 86 = RTX 30xx, 89 = RTX 40xx (fewer = faster build).
ARG CUDA_ARCHS="61;75;86;89"
ARG LLAMA_REF=master
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git curl ca-certificates python3 python3-venv python3-numpy libcurl4-openssl-dev bubblewrap \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -s /bin/bash blocky

# llama.cpp: the source stays (Blocky converts trained models with its converter), the build doesn't.
RUN git clone --depth 1 --branch "${LLAMA_REF}" https://github.com/ggml-org/llama.cpp /home/blocky/llama.cpp \
    && cd /home/blocky/llama.cpp \
    && args="-DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=ON -DGGML_RPC=ON -DGGML_NATIVE=OFF" \
    && if [ -x /usr/local/cuda/bin/nvcc ]; then \
         args="$args -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS} -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined"; \
         export PATH=/usr/local/cuda/bin:$PATH; fi \
    && cmake -S . -B build $args \
    && cmake --build build -j"$(nproc)" --target llama-server llama-bench llama-perplexity llama-quantize llama-batched-bench llama-imatrix rpc-server \
    && mkdir -p /opt/blocky/llama.cpp/bin && cp -a build/bin/. /opt/blocky/llama.cpp/bin/ \
    && rm -rf build && chown -R blocky:blocky /home/blocky/llama.cpp

COPY --chown=blocky:blocky . /opt/blocky/app
COPY --from=ui --chown=blocky:blocky /ui/dist /opt/blocky/app/frontend/dist
RUN python3 -m venv /opt/blocky/venv \
    && /opt/blocky/venv/bin/pip install --no-cache-dir -r /opt/blocky/app/backend/requirements.txt \
    && mkdir -p /opt/blocky/models /opt/blocky/data /etc/blocky \
    && chown -R blocky:blocky /opt/blocky/models /opt/blocky/data \
    && chmod +x /opt/blocky/app/scripts/start-llm.sh

# No systemd in a container: Blocky runs the chat model server itself (lab.SUPERVISE).
ENV BLOCKY_SUPERVISE_LLM=1 \
    PATH=/opt/blocky/venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
USER blocky
WORKDIR /opt/blocky/app/backend
VOLUME ["/opt/blocky/models", "/opt/blocky/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD curl -fs http://127.0.0.1:8000/api/status >/dev/null || exit 1
CMD ["uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"]
