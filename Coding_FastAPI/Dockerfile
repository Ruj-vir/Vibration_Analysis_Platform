FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN adduser --disabled-password --no-create-home vibration \
    && chown -R vibration:vibration /app
USER vibration

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
