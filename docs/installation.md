# Installation Guide

## Method 1: pip (Development)

```bash
# Clone repository
git clone https://github.com/AhmetSeyhan/Scanner2.git
cd Scanner2

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Download pre-trained deepfake weights
python download_weights.py

# Start the API server
uvicorn api:app --host 0.0.0.0 --port 8000

# Start the dashboard (separate terminal)
streamlit run dashboard.py
```

### System Dependencies

FFmpeg is required for audio extraction:

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg libgl1-mesa-glx libglib2.0-0

# macOS
brew install ffmpeg

# Windows (via chocolatey)
choco install ffmpeg
```

## Method 2: Docker (Production)

```bash
# Copy environment template
cp .env.example .env
# Edit .env with production secrets (SCANNER_SECRET_KEY, SCANNER_API_KEY)

# Development (API + Dashboard + Redis)
docker compose up -d

# Production (adds nginx reverse proxy + MinIO storage)
docker compose --profile production up -d
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| API | 8000 | FastAPI backend |
| Dashboard | 8501 | Streamlit UI |
| Redis | 6379 | Rate limiting backend |
| MinIO | 9000/9001 | Object storage (production profile) |
| Nginx | 80/443 | Reverse proxy (production profile) |

## Model Weights

Scanner supports multiple weight sources:

| Weight File | Size | Dataset | Performance |
|------------|------|---------|-------------|
| `efficientnet_b0_faceforensics.pth` | 17 MB | FaceForensics++ | Best general-purpose |
| `xception_best.pth` | 84 MB | Multi-dataset | Higher accuracy, slower |
| ImageNet (fallback) | Built-in | ImageNet | Baseline, no download needed |

Weights are stored in the `weights/` directory. The system automatically falls back to ImageNet initialization if no specialized weights are found.

## Environment Variables

See `.env.example` for the full list. Key variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `SCANNER_SECRET_KEY` | Yes | JWT signing secret |
| `SCANNER_API_KEY` | Yes | API key for service auth |
| `REDIS_URL` | No | Redis URL for rate limiting |
| `DEVICE` | No | Inference device (`cpu` or `cuda`) |

## Verification

After installation, verify the system is working:

```bash
# Check health endpoint
curl http://localhost:8000/health

# Expected response includes:
# "status": "healthy"
# "prime_hybrid": { all cores: true }
```
