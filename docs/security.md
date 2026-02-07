# Security

See [SECURITY.md](../SECURITY.md) in the project root for the full security policy.

## Quick Reference

### Authentication

Two methods supported:

1. **JWT Bearer Token**: `Authorization: Bearer <token>`
2. **API Key**: `X-API-Key: <key>`

### Scopes

| Scope | Access |
|-------|--------|
| `read` | View history, export reports |
| `write` | Run video/image analysis |
| `admin` | System stats, admin dashboard |

### Required Environment Variables

```bash
SCANNER_SECRET_KEY=<openssl rand -hex 32>
SCANNER_API_KEY=<openssl rand -hex 24>
```

### Production Checklist

- [ ] TLS 1.2+ on all endpoints
- [ ] CORS restricted to specific origins
- [ ] Unique `SCANNER_SECRET_KEY` (not default)
- [ ] Unique `SCANNER_API_KEY` (not default)
- [ ] Redis not exposed to public network
- [ ] Rate limiting enabled
- [ ] Docker containers run as non-root
