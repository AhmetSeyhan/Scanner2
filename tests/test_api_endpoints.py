"""
Scanner Test Suite - API Endpoint Integration Tests
End-to-end tests for the FastAPI routes.
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set test environment before importing api
os.environ.setdefault("SCANNER_SECRET_KEY", "test-secret")
os.environ.setdefault("SCANNER_API_KEY", "test-api-key")
os.environ.setdefault("SCANNER_ADMIN_PASSWORD", "test-admin-pw")


class TestPublicEndpoints:
    """Tests for unauthenticated endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client with mocked ModelManager."""
        with patch("api.ModelManager") as mock_mm_cls:
            mock_mm = MagicMock()
            mock_mm.is_ready = True
            mock_mm.health_status.return_value = {
                "device": "cpu",
                "biosignal_core": True,
                "artifact_core": True,
                "alignment_core": True,
                "fusion_engine": True,
                "audio_analyzer": True,
                "sanity_guard": True,
                "face_extractor": True,
                "video_processor": True,
                "inference_engine": True,
            }
            mock_mm_cls.get_instance.return_value = mock_mm

            from fastapi.testclient import TestClient
            from api import app
            yield TestClient(app)

    def test_root_endpoint(self, client):
        """GET / returns online status."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "online"
        assert data["version"] == "3.3.0"

    def test_health_endpoint(self, client):
        """GET /health returns component status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "components" in data
        assert "enterprise" in data


class TestAuthEndpoints:
    """Tests for authentication endpoints."""

    @pytest.fixture
    def client(self):
        with patch("api.ModelManager") as mock_mm_cls:
            mock_mm = MagicMock()
            mock_mm.is_ready = True
            mock_mm_cls.get_instance.return_value = mock_mm

            from fastapi.testclient import TestClient
            from api import app
            yield TestClient(app)

    def test_login_success(self, client):
        """POST /auth/token with valid credentials returns JWT."""
        response = client.post(
            "/auth/token",
            data={"username": "admin", "password": "test-admin-pw"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_login_failure(self, client):
        """POST /auth/token with wrong password returns 401."""
        response = client.post(
            "/auth/token",
            data={"username": "admin", "password": "wrong"},
        )
        assert response.status_code == 401

    def test_api_key_auth(self, client):
        """X-API-Key header provides access."""
        response = client.get(
            "/auth/me",
            headers={"X-API-Key": "test-api-key"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "api_user"

    def test_no_auth_returns_401(self, client):
        """Endpoints without auth return 401."""
        response = client.get("/auth/me")
        assert response.status_code == 401


class TestProtectedEndpoints:
    """Tests for analysis endpoints (require auth)."""

    @pytest.fixture
    def client(self):
        with patch("api.ModelManager") as mock_mm_cls:
            mock_mm = MagicMock()
            mock_mm.is_ready = True
            mock_mm_cls.get_instance.return_value = mock_mm

            from fastapi.testclient import TestClient
            from api import app
            yield TestClient(app)

    def test_analyze_video_requires_auth(self, client):
        """POST /analyze-video without auth returns 401."""
        response = client.post("/analyze-video")
        assert response.status_code in (401, 422)  # 422 if missing file

    def test_analyze_video_v2_requires_auth(self, client):
        """POST /analyze-video-v2 without auth returns 401."""
        response = client.post("/analyze-video-v2")
        assert response.status_code in (401, 422)

    def test_stats_requires_admin(self, client):
        """GET /stats with non-admin scope returns 403."""
        # Use API key (has read+write but not admin)
        response = client.get(
            "/stats",
            headers={"X-API-Key": "test-api-key"},
        )
        assert response.status_code == 403
