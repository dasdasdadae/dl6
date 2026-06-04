FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY predict.py .
COPY best_model.pkl .

ENV TORCH_SKIP_WEIGHTS_ONLY_UNPICKLE=1

ENTRYPOINT ["python", "predict.py"]