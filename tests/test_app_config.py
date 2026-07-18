import os
import unittest

from app import get_server_config


class ServerConfigTests(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("PORT", None)
        os.environ.pop("FLASK_HOST", None)
        os.environ.pop("FLASK_DEBUG", None)

    def test_default_server_config(self):
        config = get_server_config()
        self.assertEqual(config["host"], "0.0.0.0")
        self.assertEqual(config["port"], 5000)
        self.assertFalse(config["debug"])

    def test_env_override_server_config(self):
        os.environ["PORT"] = "8000"
        os.environ["FLASK_HOST"] = "127.0.0.1"
        os.environ["FLASK_DEBUG"] = "true"

        config = get_server_config()
        self.assertEqual(config["host"], "127.0.0.1")
        self.assertEqual(config["port"], 8000)
        self.assertTrue(config["debug"])


if __name__ == "__main__":
    unittest.main()
