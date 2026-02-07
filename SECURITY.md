# Security Policy

## Reporting Vulnerabilities

If you discover a security vulnerability in Scanner, please report it responsibly:

- **Email**: security@scanner.ai
- **Subject**: `[SECURITY] <Brief Description>`
- **PGP Key**: Available upon request

We will acknowledge receipt within 24 hours and provide an initial assessment within 72 hours. Please do not open public issues for security vulnerabilities.

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 3.2.x   | Yes (current)      |
| 3.0.x   | Security fixes only |
| < 3.0   | No                 |

## Security Architecture

### Authentication

Scanner uses a dual authentication model:

1. **JWT Bearer Tokens** (recommended for production)
   - Algorithm: HS256 (HMAC-SHA256)
   - Token lifetime: Configurable (default 24 hours)
   - Scopes: `read`, `write`, `admin`
   - Tokens are stateless; revocation requires secret key rotation

2. **API Key Authentication** (for service-to-service)
   - Header: `X-API-Key`
   - Keys are compared in constant time to prevent timing attacks
   - Scoped to `read` + `write` by default

**Production Requirements:**
- Set `SCANNER_SECRET_KEY` to a cryptographically random value (minimum 32 bytes)
- Set `SCANNER_API_KEY` to a unique key per deployment
- Rotate secrets on a regular schedule
- Never commit secrets to version control

### Rate Limiting

- **Library**: slowapi (based on limits)
- **Backend**: Redis (persistent across restarts)
- **Default**: 10 requests per minute per IP address
- **Configuration**: Via `RATE_LIMIT` environment variable
- Rate limit headers included in responses: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`

### Input Validation

- **File type whitelist**: Only `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm` (video) and `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp` (image) are accepted
- **InputSanityGuard**: Pre-analysis adversarial detection
  - Frame sequence consistency checks
  - Resolution consistency validation
  - Adversarial gradient pattern detection (Sobel + FFT)
  - Content integrity verification (NaN/Inf/extreme values)
- **File size**: Enforce maximum upload size via reverse proxy (recommended: 500 MB)

### Data Privacy

Scanner is designed with privacy-first principles:

- **No persistent video storage**: Uploaded files are processed in temporary directories and deleted immediately after analysis
- **Temp file cleanup**: `shutil.rmtree()` in `finally` blocks ensures cleanup even on errors
- **Analysis results only**: Only metadata and scores are stored, never raw video content
- **History retention**: SQLite history limited to last 100 entries with automatic cleanup
- **No telemetry**: Scanner does not phone home or collect usage data

### GDPR / KVKK Compliance Notes

For deployments processing EU/TR personal data:

| Requirement | Scanner Implementation |
|-------------|----------------------|
| Data minimization | Only detection scores stored, no raw media |
| Right to deletion | `DELETE /history/{id}` endpoint; `clear_history()` API |
| Purpose limitation | Analysis-only; no profiling or tracking |
| Storage limitation | Auto-cleanup after 100 entries |
| Security of processing | TLS required in production; JWT auth; rate limiting |
| Data portability | JSON API responses; PDF export |

**Recommendations for GDPR compliance:**
- Deploy behind TLS/HTTPS (see nginx config in `docker-compose.yml`)
- Implement proper user consent workflows in your integration layer
- Log and audit all access to analysis endpoints
- Maintain a Record of Processing Activities (ROPA) document

### Network Security

**Production Deployment Checklist:**

- [ ] TLS 1.2+ enforced on all endpoints
- [ ] CORS restricted to specific origins (not `*`)
- [ ] API behind reverse proxy (nginx/Traefik)
- [ ] Redis not exposed to public network
- [ ] MinIO (if used) behind internal network only
- [ ] Docker containers run as non-root user (`scanner`)
- [ ] Health check endpoints (`/`, `/health`) do not expose sensitive information
- [ ] Rate limiting enabled with Redis backend

### Secret Management

Scanner reads configuration from environment variables. Never hardcode secrets.

**Required Environment Variables:**

| Variable | Description | Example |
|----------|-------------|---------|
| `SCANNER_SECRET_KEY` | JWT signing key | `openssl rand -hex 32` |
| `SCANNER_API_KEY` | API key for service auth | `openssl rand -hex 24` |
| `REDIS_URL` | Redis connection string | `redis://redis:6379/0` |
| `S3_ENDPOINT_URL` | MinIO/S3 endpoint | `http://minio:9000` |
| `AWS_ACCESS_KEY_ID` | S3 access key | (deployment-specific) |
| `AWS_SECRET_ACCESS_KEY` | S3 secret key | (deployment-specific) |

**Example `.env` file:**

```bash
# Copy to .env and fill in production values
SCANNER_SECRET_KEY=<output of: openssl rand -hex 32>
SCANNER_API_KEY=<output of: openssl rand -hex 24>
REDIS_URL=redis://localhost:6379/0
```

### Dependency Security

- CI/CD includes Trivy vulnerability scanning on every push
- Dependencies pinned to known-good versions in `requirements.txt`
- Docker image built from `python:3.12-slim` (minimal attack surface)
- No unnecessary system packages installed in container

### Threat Model

| Threat | Mitigation |
|--------|-----------|
| Adversarial input (model evasion) | InputSanityGuard pre-screening |
| Authentication bypass | JWT + API key dual auth; scope enforcement |
| Brute force login | Rate limiting (10 req/min); token expiry |
| Man-in-the-middle | TLS required in production |
| Denial of service | Rate limiting; resource limits in Docker |
| Data exfiltration | No raw media storage; scope-based access control |
| Container escape | Non-root user; read-only volume mounts for weights |
| Supply chain attack | Trivy scanning; pinned dependencies |
