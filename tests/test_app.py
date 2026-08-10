"""
Unit tests for Flask API routes in backend/app.py.

Uses Flask's built-in test client — no real HTTP server required.
The pipeline worker thread is mocked so tests are fast and deterministic.
"""

import io
import json
import os
import unittest
from unittest.mock import patch

# Provide env vars before importing app so dotenv doesn't override them
os.environ.setdefault("MQTT_BROKER_HOST", "localhost")
os.environ.setdefault("MQTT_BROKER_PORT", "1883")

from backend.app import app


def _mp4_upload(client, filename="traffic.mp4", content=b"\x00" * 64):
    """Helper: POST a fake file to /api/upload via Flask test client."""
    data = {"video": (io.BytesIO(content), filename)}
    return client.post(
        "/api/upload",
        data=data,
        content_type="multipart/form-data",
    )


class TestFlaskRoutes(unittest.TestCase):
    """Test suite for Flask API endpoints."""

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    # ── GET / ──────────────────────────────────────────────────────────

    def test_index_returns_200(self):
        """Root route must return the dashboard HTML page."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Edge-CV", resp.data)

    # ── POST /api/upload ───────────────────────────────────────────────

    def test_upload_no_file_field_returns_400(self):
        """Upload with no 'video' field must return 400."""
        resp = self.client.post("/api/upload", data={})
        self.assertEqual(resp.status_code, 400)
        body = json.loads(resp.data)
        self.assertIn("error", body)

    def test_upload_unsupported_format_returns_415(self):
        """Upload of a .txt file must return 415 Unsupported Media Type."""
        resp = _mp4_upload(self.client, filename="report.txt")
        self.assertEqual(resp.status_code, 415)
        body = json.loads(resp.data)
        self.assertIn("error", body)

    @patch("backend.app._run_pipeline")
    def test_upload_valid_mp4_returns_202_with_job_id(self, mock_run):
        """Valid MP4 upload must return 202 with a job_id."""
        mock_run.return_value = None
        resp = _mp4_upload(self.client, filename="traffic.mp4")
        self.assertEqual(resp.status_code, 202)
        body = json.loads(resp.data)
        self.assertIn("job_id", body)
        self.assertIn("filename", body)
        self.assertEqual(body["filename"], "traffic.mp4")

    @patch("backend.app._run_pipeline")
    def test_upload_valid_avi_returns_202(self, mock_run):
        """AVI format must also be accepted."""
        mock_run.return_value = None
        resp = _mp4_upload(self.client, filename="clip.avi")
        self.assertEqual(resp.status_code, 202)

    # ── GET /api/status/<job_id> ────────────────────────────────────────

    def test_status_unknown_job_returns_404(self):
        """Status for a non-existent job_id must return 404."""
        resp = self.client.get("/api/status/nonexistent")
        self.assertEqual(resp.status_code, 404)

    @patch("backend.app._run_pipeline")
    def test_status_known_job_returns_200(self, mock_run):
        """Status for a valid job must return 200 with status field."""
        mock_run.return_value = None
        upload = _mp4_upload(self.client, filename="test.mp4")
        self.assertEqual(upload.status_code, 202, msg=upload.data.decode())
        job_id = json.loads(upload.data)["job_id"]

        resp = self.client.get(f"/api/status/{job_id}")
        self.assertEqual(resp.status_code, 200)
        body = json.loads(resp.data)
        self.assertIn("status", body)
        self.assertIn("job_id", body)
        self.assertEqual(body["job_id"], job_id)

    # ── GET /api/stream/<job_id> ────────────────────────────────────────

    def test_stream_unknown_job_returns_404(self):
        """SSE stream for a non-existent job must return 404."""
        resp = self.client.get("/api/stream/nonexistent")
        self.assertEqual(resp.status_code, 404)


if __name__ == "__main__":
    unittest.main()
