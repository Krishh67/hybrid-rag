import pytest
from unittest.mock import MagicMock, patch
from retrieval.query_rewriter import QueryRewriter
from google.api_core.exceptions import ResourceExhausted

def test_query_rewriter_parsing():
    rewriter = QueryRewriter()
    # Ensure it's active for the test
    rewriter.client_ready = True
    
    original = "What are the grading rules?"
    
    # Test JSON parsing
    json_response = '```json\n["What are the grading criteria?", "How are grades assigned?", "What is the assessment policy?"]\n```'
    parsed = rewriter._parse_response(json_response, original)
    assert len(parsed) == 4
    assert parsed[0] == original
    assert parsed[1] == "What are the grading criteria?"
    
    # Test text fallback
    text_response = "- What are the grading criteria?\n- How are grades assigned?\n- What is the assessment policy?\n- Extra query"
    parsed2 = rewriter._parse_response(text_response, original)
    assert len(parsed2) == 4
    assert parsed2[0] == original
    assert parsed2[1] == "What are the grading criteria?"

@patch("retrieval.query_rewriter.genai.GenerativeModel")
def test_query_rewriter_fallback(mock_model_class):
    rewriter = QueryRewriter()
    rewriter.client_ready = True
    
    # We want the first model to raise ResourceExhausted, and the second to succeed
    mock_instance_1 = MagicMock()
    mock_instance_1.generate_content.side_effect = ResourceExhausted("Rate limit hit")
    
    mock_instance_2 = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '["q1", "q2", "q3"]'
    mock_instance_2.generate_content.return_value = mock_response
    
    # side_effect list returns the instances in order
    mock_model_class.side_effect = [mock_instance_1, mock_instance_2]
    
    results = rewriter.rewrite("original")
    
    assert len(results) == 4
    assert results[0] == "original"
    assert results[1] == "q1"
    
    # Ensure both models were attempted
    assert mock_model_class.call_count == 2
