import unittest
from pathlib import Path
import shutil
import tempfile

from fastapi.testclient import TestClient

import rag_app.api as api
from rag_app.config import AppConfig
from rag_app.pipeline import RAGPipeline


class ApiTests(unittest.TestCase):
    def test_health_and_stats(self):
        client = TestClient(api.app)

        health = client.get("/api/health")
        stats = client.get("/api/stats")

        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ok")
        self.assertEqual(stats.status_code, 200)
        self.assertIn("chunks", stats.json())

    def test_files_api_lists_details_and_deletes_upload(self):
        root = Path(tempfile.mkdtemp())
        original_pipeline = api.pipeline
        try:
            api.pipeline = RAGPipeline(AppConfig.from_root(root))
            client = TestClient(api.app)

            upload = client.post(
                "/api/upload",
                files={"file": ("agent.md", b"AI Agent uses tools and memory.", "text/markdown")},
            )
            files = client.get("/api/files")
            file_item = files.json()["files"][0]
            detail = client.get(f"/api/files/{file_item['id']}")
            delete = client.delete(f"/api/files/{file_item['id']}")
            files_after_delete = client.get("/api/files")

            self.assertEqual(upload.status_code, 200)
            self.assertEqual(files.status_code, 200)
            self.assertEqual(file_item["filename"], "agent.md")
            self.assertEqual(detail.status_code, 200)
            self.assertEqual(detail.json()["chunks_count"], 1)
            self.assertEqual(delete.status_code, 200)
            self.assertEqual(delete.json()["status"], "deleted")
            self.assertEqual(files_after_delete.json()["files"], [])
        finally:
            api.pipeline = original_pipeline
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
