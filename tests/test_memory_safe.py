import unittest
from unittest.mock import MagicMock, patch

import app
import model_utils


class MemorySafeTests(unittest.TestCase):
    def test_health_endpoint(self):
        response = app.app.test_client().get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"status": "ok"})

    def test_similarity_falls_back_without_api_client(self):
        with patch.object(model_utils, "_get_openai_client", return_value=None):
            score = model_utils.compute_similarity("python flask", "python flask mongodb")
        self.assertGreater(score, 0)
        self.assertLessEqual(score, 1)

    def test_dashboard_projects_and_limits_records(self):
        cursor = MagicMock()
        cursor.sort.return_value = cursor
        cursor.limit.return_value = [{
            "_id": "abc",
            "name": "sample.pdf",
            "score": 80,
            "similarity": 75,
            "skills": ["python"],
            "matched": ["python"],
            "missing": [],
            "feedback": "Good match.",
        }]
        collection = MagicMock()
        collection.find.return_value = cursor

        with patch.object(app, "get_candidate_collection", return_value=collection):
            response = app.app.test_client().get("/dashboard")

        self.assertEqual(response.status_code, 200)
        collection.find.assert_called_once()
        cursor.limit.assert_called_once_with(app.DASHBOARD_LIMIT)


if __name__ == "__main__":
    unittest.main()
