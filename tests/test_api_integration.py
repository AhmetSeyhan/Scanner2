"""
Integration tests for FastAPI endpoints.
Tests authentication, health checks, and error handling.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def client():
    """Create test client."""
    try:
        from fastapi.testclient import TestClient

        from api import app
        return TestClient(app)
    except Exception:
        pytest.skip("FastAPI test client not available (startup may require MediaPipe)")


@pytest.fixture
def auth_headers(client):
    """Get auth headers with valid token."""
    admin_pw = os.getenv("SCANNER_ADMIN_PASSWORD", "test-admin-pw")
    response = client.post(
        "/auth/token",
        data={"username": "admin", "password": admin_pw}
    )
    if response.status_code != 200:
        pytest.skip("Auth not available")
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


class TestPublicEndpoints:
    """Test public (unauthenticated) endpoints."""

    def test_root(self, client):
        """Test root health check."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "online"
        assert "version" in data

    def test_health(self, client):
        """Test detailed health check."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("healthy", "degraded")
        assert "components" in data
        assert "enterprise" in data

    def test_docs_available(self, client):
        """Test that OpenAPI docs are available."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_spec(self, client):
        """Test that OpenAPI JSON spec is generated."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        spec = response.json()
        assert spec["info"]["title"] == "Scanner API"
        assert spec["info"]["version"] != ""


class TestAuthentication:
    """Test authentication endpoints."""

    def test_login_success(self, client):
        """Test successful login."""
        admin_pw = os.getenv("SCANNER_ADMIN_PASSWORD", "test-admin-pw")
        response = client.post(
            "/auth/token",
            data={"username": "admin", "password": admin_pw}
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_login_invalid_password(self, client):
        """Test login with wrong password."""
        response = client.post(
            "/auth/token",
            data={"username": "admin", "password": "wrong"}
        )
        assert response.status_code == 401

    def test_login_invalid_user(self, client):
        """Test login with non-existent user."""
        response = client.post(
            "/auth/token",
            data={"username": "nobody", "password": "nothing"}
        )
        assert response.status_code == 401

    def test_get_me(self, client, auth_headers):
        """Test getting current user info."""
        response = client.get("/auth/me", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "admin"

    def test_protected_without_auth(self, client):
        """Test accessing protected endpoint without auth."""
        response = client.get("/stats")
        assert response.status_code == 401

    def test_api_key_auth(self, client):
        """Test API key authentication."""
        api_key = os.getenv("SCANNER_API_KEY", "test-api-key")
        response = client.get(
            "/auth/me",
            headers={"X-API-Key": api_key}
        )
        assert response.status_code == 200

    def test_invalid_api_key(self, client):
        """Test invalid API key."""
        response = client.get(
            "/auth/me",
            headers={"X-API-Key": "invalid-key"}
        )
        assert response.status_code == 401


class TestAnalysisEndpoints:
    """Test analysis endpoint validation (without actual video processing)."""

    def test_analyze_video_no_file(self, client, auth_headers):
        """Test video analysis without file."""
        response = client.post("/analyze-video", headers=auth_headers)
        assert response.status_code == 422  # Validation error

    def test_analyze_video_wrong_type(self, client, auth_headers):
        """Test video analysis with wrong file type."""
        response = client.post(
            "/analyze-video",
            headers=auth_headers,
            files={"file": ("test.txt", b"not a video", "text/plain")}
        )
        assert response.status_code == 400

    def test_analyze_image_wrong_type(self, client, auth_headers):
        """Test image analysis with wrong file type."""
        response = client.post(
            "/analyze-image",
            headers=auth_headers,
            files={"file": ("test.pdf", b"not an image", "application/pdf")}
        )
        assert response.status_code == 400

    def test_scope_enforcement(self, client):
        """Test that viewer (read-only) cannot analyze."""
        response = client.post(
            "/auth/token",
            data={"username": "viewer", "password": "test-viewer-pw"}
        )
        if response.status_code != 200:
            pytest.skip("Viewer auth not available")
        token = response.json()["access_token"]

        response = client.post(
            "/analyze-video",
            headers={"Authorization": f"Bearer {token}"},
            files={"file": ("test.mp4", b"fake", "video/mp4")}
        )
        assert response.status_code == 403


class TestHistoryEndpoints:
    """Test history endpoints."""

    def test_history_requires_auth(self, client):
        """Test history requires authentication."""
        response = client.get("/history")
        assert response.status_code == 401

    def test_history_not_found(self, client, auth_headers):
        """Test non-existent session."""
        response = client.get("/history/nonexistent-id", headers=auth_headers)
        # May be 404 or 503 depending on whether history manager is available
        assert response.status_code in [404, 503]


class TestAdminEndpoints:
    """Test admin endpoints."""

    def test_stats_requires_admin(self, client):
        """Test stats requires admin scope."""
        # Login as viewer (read only)
        response = client.post(
            "/auth/token",
            data={"username": "viewer", "password": "test-viewer-pw"}
        )
        if response.status_code != 200:
            pytest.skip("Viewer auth not available")
        token = response.json()["access_token"]

        response = client.get(
            "/stats",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 403

    def test_stats_with_admin(self, client, auth_headers):
        """Test stats with admin credentials."""
        response = client.get("/stats", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "engine" in data
        assert "components" in data
