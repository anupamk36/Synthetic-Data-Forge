import unittest
from unittest.mock import patch, MagicMock
import json
import polars as pl
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.llm_logic import LLMLogicEngine
from core.generator import ForgeEngine

class TestLLMGeneration(unittest.TestCase):

    def setUp(self):
        self.llm = LLMLogicEngine()
        self.engine = ForgeEngine()
        self.schema = {"name": "String", "city": "String", "state": "String"}

    @patch('requests.get')
    @patch('requests.post')
    def test_llm_batch_generation(self, mock_post, mock_get):
        # Mock Ollama availability
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"models": [{"name": "llama3"}]}

        # Mock Ollama generation response
        mock_records = [
            {"name": "Alice", "city": "New York", "state": "NY"},
            {"name": "Bob", "city": "Los Angeles", "state": "CA"}
        ]
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "response": json.dumps(mock_records)
        }

        # Test llm_logic.generate_data
        records = self.llm.generate_data(self.schema, 2)
        
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["city"], "New York")
        self.assertEqual(mock_post.call_count, 1)

    @patch('core.llm_logic.LLMLogicEngine.is_available')
    @patch('core.llm_logic.LLMLogicEngine.generate_data')
    def test_forge_engine_integration(self, mock_gen_data, mock_available):
        mock_available.return_value = True
        mock_records = [
            {"name": "Alice", "city": "New York", "state": "NY"},
            {"name": "Bob", "city": "Los Angeles", "state": "CA"}
        ]
        mock_gen_data.return_value = mock_records

        # Test ForgeEngine with use_llm=True
        df = self.engine.generate_records(self.schema, 2, use_llm=True, llm_engine=self.llm)
        
        self.assertIsInstance(df, pl.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertEqual(df["city"][0], "New York")
        mock_gen_data.assert_called_once()

    @patch('core.llm_logic.LLMLogicEngine.generate_data')
    def test_forge_engine_fallback(self, mock_gen_data):
        # Mock LLM failure (returns empty list)
        mock_gen_data.return_value = []

        # Should fallback to Faker
        df = self.engine.generate_records(self.schema, 5, use_llm=True, llm_engine=self.llm)
        
        self.assertEqual(len(df), 5)
        # Verify it's not empty and columns exist
        self.assertIn("city", df.columns)
        self.assertTrue(len(df["city"]) == 5)

if __name__ == '__main__':
    unittest.main()
