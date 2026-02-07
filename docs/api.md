# API Reference

Scanner exposes a RESTful API via FastAPI. Interactive documentation is available at `/docs` (Swagger) and `/redoc` (ReDoc) when the server is running.

## Authentication

All protected endpoints require one of:

### JWT Bearer Token

```bash
# Obtain token
curl -X POST http://localhost:8000/auth/token \
  -d "username=admin&password=scanner2026"

# Use in requests
curl -H "Authorization: Bearer <token>" http://localhost:8000/history
```

### API Key

```bash
curl -H "X-API-Key: <your-api-key>" http://localhost:8000/history
```

### Scopes

| Scope | Capabilities |
|-------|-------------|
| `read` | View history, export reports |
| `write` | Run analyses (video, image) |
| `admin` | System stats, admin dashboard |

## Endpoints

### Public

#### `GET /`

Health check. Returns API status and version.

#### `GET /health`

Detailed component health including PRIME HYBRID core status and enterprise feature availability.

### Authentication

#### `POST /auth/token`

Request body (form-encoded): `username`, `password`

Returns: `{ "access_token": "...", "token_type": "bearer", "expires_in": 86400 }`

#### `GET /auth/me`

Returns current user info. Requires any valid authentication.

### Analysis

#### `POST /analyze-video-v2`

**Full PRIME HYBRID analysis.** This is the primary analysis endpoint.

- **Auth**: `write` scope
- **Input**: Multipart file upload (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`)
- **Response**: Complete forensic analysis with verdict, per-core scores, transparency report

#### `POST /analyze-video`

Basic video analysis using EfficientNet-B0 backbone only (without PRIME HYBRID cores).

- **Auth**: `write` scope
- **Input**: Multipart file upload

#### `POST /analyze-image`

Single image analysis.

- **Auth**: `write` scope
- **Input**: Multipart file upload (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`)

### History & Export

#### `GET /history`

Query parameters: `limit` (default 20, max 100)

Returns recent scan entries.

#### `GET /history/stats`

Aggregate statistics: total scans, verdict distribution, average integrity score.

#### `GET /history/{session_id}`

Single scan result by session ID.

#### `POST /export/pdf/{session_id}`

Generate and download PDF forensic report for a completed scan.

### Admin

#### `GET /stats`

System statistics including model info, component status, and scan history.

- **Auth**: `admin` scope

#### `GET /admin/dashboard`

Admin dashboard with system metrics, recent activity, and component status.

- **Auth**: `admin` scope

## Rate Limiting

Default: 10 requests per minute per IP address.

Rate limit headers are included in all responses:

- `X-RateLimit-Limit`: Maximum requests per window
- `X-RateLimit-Remaining`: Requests remaining
- `X-RateLimit-Reset`: Window reset time

## Error Responses

All errors follow the format:

```json
{
  "detail": "Error description"
}
```

| Status Code | Meaning |
|-------------|---------|
| 400 | Bad request (invalid file type, corrupted input) |
| 401 | Authentication required or invalid credentials |
| 403 | Insufficient permissions (missing scope) |
| 404 | Resource not found |
| 429 | Rate limit exceeded |
| 500 | Internal server error |
| 503 | Service unavailable (component not initialized) |
