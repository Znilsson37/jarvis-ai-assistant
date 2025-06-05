import unittest
from modules.grok_client import GrokClient
from unittest.mock import patch
import json

class TestGrokClient(unittest.TestCase):
    def setUp(self):
        self.client = GrokClient(api_key="test-key")

    @patch('requests.Session.post')
    def test_basic_connection(self, mock_post):
        # Setup mock response
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "Connection successful"}}]
        }
        mock_post.return_value.status_code = 200
        
        # Test basic connection
        result = self.client.test_connection()
        self.assertTrue(result)
        
        # Verify correct endpoint
        self.assertEqual(self.client.endpoint, "https://api.x.ai/v1/chat/completions")

    @patch('requests.Session.post')
    def test_live_search_news(self, mock_post):
        # Setup mock response
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "Test news content"}}]
        }
        mock_post.return_value.status_code = 200
        
        # Test news with live search
        result = self.client.get_news(
            query="test news",
            search_mode="auto",
            return_citations=True,
            from_date="2024-01-01",
            to_date="2024-12-31",
            max_search_results=10,
            sources=[{"type": "news"}]
        )
        
        # Verify the request payload
        called_payload = mock_post.call_args[1]['json']
        self.assertIn("search_parameters", called_payload)
        self.assertEqual(called_payload["search_parameters"]["mode"], "auto")
        self.assertEqual(called_payload["search_parameters"]["max_search_results"], 10)

    @patch('requests.Session.post')
    def test_error_handling(self, mock_post):
        # Test invalid API key
        mock_post.return_value.status_code = 401
        mock_post.return_value.raise_for_status.side_effect = Exception("Invalid API key")
        
        result = self.client.ask("test prompt")
        self.assertIn("[Grok API Error]", result)

    @patch('requests.Session.post')
    def test_chat_with_search(self, mock_post):
        # Setup mock response
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        mock_post.return_value.status_code = 200
        
        # Test chat with search parameters
        search_params = {
            "mode": "auto",
            "return_citations": True,
            "sources": [{"type": "web"}]
        }
        
        result = self.client.chat(
            messages=[{"role": "user", "content": "test"}],
            search_parameters=search_params
        )
        
        # Verify search parameters in request
        called_payload = mock_post.call_args[1]['json']
        self.assertIn("search_parameters", called_payload)
        self.assertEqual(called_payload["search_parameters"], search_params)

if __name__ == '__main__':
    unittest.main()
