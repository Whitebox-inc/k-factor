# tests/test_kay_agent.py

import pytest
from unittest.mock import patch, MagicMock
import json

# Import classes from kay_agent.py
from kay.kay2 import KayAgent
from langchain.llms import AI21
from kay.tool_module import tools as loaded_tools


@pytest.fixture
def mock_llm():
    with patch("kay_agent.AI21") as MockLLM:
        mock_llm_instance = MockLLM.return_value
        mock_llm_instance.run.return_value = "Test LLM response"
        llm = AI21(model="j2-ultra", api_key="test-api-key")
        yield llm


@pytest.fixture
def kay_agent_instance(mock_llm):
    agent = KayAgent(llm=mock_llm, tools=loaded_tools)
    return agent


def test_kay_handle_user_input(kay_agent_instance):
    with patch.object(kay_agent_instance.agent, "run") as mock_agent_run:
        mock_agent_run.return_value = '{"status_code": 200, "data": {"key": "value"}}'
        user_input = "Test user input"
        response = kay_agent_instance.handle_user_input(user_input)
        assert response == '{"status_code": 200, "data": {"key": "value"}}\n\n'
        mock_agent_run.assert_called_once_with(user_input)


def test_kay_handle_user_input_with_visualization(kay_agent_instance):
    # Mock agent's run method to return JSON suitable for visualization
    mock_response = json.dumps(
        {
            "status_code": 200,
            "data": {"Category A": 10, "Category B": 20, "Category C": 30},
        }
    )
    with patch.object(kay_agent_instance.agent, "run") as mock_agent_run:
        mock_agent_run.return_value = mock_response
        user_input = "Show me the data distribution."
        response = kay_agent_instance.handle_user_input(user_input)
        # Check if the response contains the JSON data and an embedded image
        assert '"Category A": 10' in response
        assert '<img src="data:image/png;base64,' in response


def test_kay_handle_user_input_with_non_json_response(kay_agent_instance):
    # Mock agent's run method to return plain text
    mock_response = "Plain text response from tool."
    with patch.object(kay_agent_instance.agent, "run") as mock_agent_run:
        mock_agent_run.return_value = mock_response
        user_input = "Provide a plain text response."
        response = kay_agent_instance.handle_user_input(user_input)
        assert response == "Plain text response from tool."


def test_kay_handle_user_input_error(kay_agent_instance):
    # Mock agent's run method to raise an exception
    with patch.object(
        kay_agent_instance.agent, "run", side_effect=Exception("Agent error")
    ):
        user_input = "Trigger an error."
        response = kay_agent_instance.handle_user_input(user_input)
        assert response == "Error: Agent error"
