"""
Unit tests for Flask API routes in backend/app.py.

Uses Flask's built-in test client — no real HTTP server required.
The pipeline worker thread is mocked so tests are fast and deterministic.
"""

import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

# Provide env vars before importing app so dotenv doesn't override them
os.environ.setdefault("MQTT_BROKER_HOST", "localhost")
os.environ.setdefault("MQTT_BROKER_PORT", "1883")

from backend.app import app, _jobs, _jobs_lock


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
        self.temp_dir = tempfile.TemporaryDirectory()
        temp_root = Path(self.temp_dir.name)
        self.upload_dir = temp_root / "videos"
        self.output_dir = temp_root / "outputs"
        self.upload_dir.mkdir()
        self.output_dir.mkdir()
        self.directory_patches = [
            patch("backend.app.UPLOAD_DIR", self.upload_dir),
            patch("backend.app.OUTPUT_DIR", self.output_dir),
        ]
        for directory_patch in self.directory_patches:
            directory_patch.start()
        self.client = app.test_client()

    def tearDown(self):
        for directory_patch in reversed(self.directory_patches):
            directory_patch.stop()
        self.temp_dir.cleanup()

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
        self.assertIn("source_url", body)
        self.assertEqual(body["filename"], "traffic.mp4")

        raw_url = body["source_url"].replace("/media/play/", "/media/")
        media = self.client.get(raw_url)
        self.assertEqual(media.status_code, 200)
        self.assertEqual(media.data, b"\x00" * 64)
        media.close()

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

    def test_video_library_returns_uploads_and_outputs(self):
        """The UI media library endpoint must expose both collections."""
        resp = self.client.get("/api/videos")
        self.assertEqual(resp.status_code, 200)
        body = json.loads(resp.data)
        self.assertIn("uploads", body)
        self.assertIn("outputs", body)
        self.assertTrue(all("url" in item for item in body["outputs"]))

    def test_unknown_media_collection_returns_404(self):
        resp = self.client.get("/media/private/video.mp4")
        self.assertEqual(resp.status_code, 404)

    @patch("backend.app._run_pipeline")
    def test_live_preview_returns_only_new_annotated_frame(self, mock_run):
        mock_run.return_value = None
        upload = _mp4_upload(self.client)
        job_id = json.loads(upload.data)["job_id"]

        self.assertEqual(self.client.get(f"/api/preview/{job_id}").status_code, 204)
        with _jobs_lock:
            _jobs[job_id]["preview"] = b"jpeg-data"
            _jobs[job_id]["preview_version"] = 7
            _jobs[job_id]["preview_frame"] = 12

        preview = self.client.get(f"/api/preview/{job_id}?since=0")
        self.assertEqual(preview.status_code, 200)
        self.assertEqual(preview.content_type, "image/jpeg")
        self.assertEqual(preview.headers["X-Preview-Version"], "7")
        self.assertEqual(preview.headers["X-Preview-Frame"], "12")
        self.assertEqual(preview.data, b"jpeg-data")
        self.assertEqual(
            self.client.get(f"/api/preview/{job_id}?since=7").status_code, 204
        )

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
