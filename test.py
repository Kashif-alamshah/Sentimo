import unittest
from sentimo import app
from flask import json

class FlaskAppTestCase(unittest.TestCase):
    
    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    def test_home_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Sentimo", response.data)

    def test_about_page(self):
        response = self.client.get("/about")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"About Sentimo", response.data)

    def test_analyze_valid_input(self):
        data = {
            "text": "I love this product!",
            "language": "en"
        }
        response = self.client.post("/analyze", data=data)
        self.assertEqual(response.status_code, 200)
        response_json = json.loads(response.data)
        self.assertIn("sentiment", response_json)
        self.assertIsInstance(response_json["sentiment"], (int, float))

    def test_analyze_missing_text(self):
        data = {
            "text": "",
            "language": "en"
        }
        response = self.client.post("/analyze", data=data)
        self.assertEqual(response.status_code, 400)
        response_json = json.loads(response.data)
        self.assertEqual(response_json["error"], "No text provided")

    def test_analyze_non_english_input(self):
        data = {
            "text": "¡Me encanta este producto!",
            "language": "es"
        }
        response = self.client.post("/analyze", data=data)
        self.assertEqual(response.status_code, 200)
        response_json = json.loads(response.data)
        self.assertIn("sentiment", response_json)
        self.assertIsInstance(response_json["sentiment"], (int, float))

    def test_invalid_route(self):
        response = self.client.get("/invalidroute")
        self.assertEqual(response.status_code, 404)

if __name__ == "__main__":
    unittest.main(verbosity=2)
