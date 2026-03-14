FROM python:3.11-slim

# システム依存パッケージ（PyAV用）
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# モデルキャッシュ用ディレクトリ
ENV HF_HOME=/app/.cache/huggingface

EXPOSE 8888

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8888", "--timeout", "600", "--workers", "1"]
