# Deployment Guide

## Docker Compose (Recommended)

### Development

```bash
cp .env.example .env
# Edit .env with your secrets

docker compose up -d
```

This starts: API (port 8000), Dashboard (port 8501), Redis (port 6379).

### Production

```bash
docker compose --profile production up -d
```

Adds: Nginx reverse proxy (ports 80/443) and MinIO storage (ports 9000/9001).

### Scaling API Workers

```yaml
# docker-compose.override.yml
services:
  api:
    deploy:
      replicas: 4
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
```

```bash
docker compose up -d --scale api=4
```

## Manual Deployment

### Single Server

```bash
pip install -r requirements.txt

# Production server with multiple workers
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

### Behind Nginx

Example nginx configuration for TLS termination:

```nginx
upstream scanner_api {
    server 127.0.0.1:8000;
}

server {
    listen 443 ssl http2;
    server_name api.scanner.ai;

    ssl_certificate /etc/ssl/scanner.crt;
    ssl_certificate_key /etc/ssl/scanner.key;

    client_max_body_size 500M;

    location / {
        proxy_pass http://scanner_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Monitoring

### Health Checks

```bash
# Basic health
curl http://localhost:8000/

# Detailed component status
curl http://localhost:8000/health
```

### Structured Logging

Set `LOG_LEVEL=INFO` in your environment. Scanner uses Python's standard logging module.

### Prometheus Metrics

Add the `prometheus-fastapi-instrumentator` package for metrics:

```bash
pip install prometheus-fastapi-instrumentator
```

The `/metrics` endpoint exposes request latency, throughput, and error rates.

## Offline Operation

Scanner can operate fully offline:

1. Pre-download model weights: `python download_weights.py`
2. Pre-download MediaPipe model: runs automatically on first start
3. Set `REDIS_URL` to a local Redis instance
4. No external API calls are made during analysis

## Backup and Recovery

### Database

Scan history is stored in SQLite at `~/.scanner/history.db`:

```bash
# Backup
cp ~/.scanner/history.db ~/.scanner/history.db.bak

# Restore
cp ~/.scanner/history.db.bak ~/.scanner/history.db
```

### Model Weights

Store weight files in version-controlled or backed-up storage. The `weights/` directory should be mounted as a read-only volume in Docker.
