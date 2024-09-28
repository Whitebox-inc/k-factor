# tests/test_tool_module.py

import pytest
from unittest.mock import patch, MagicMock
import json
import os

# Import functions from tool_module.py
from kay.tool_module import load_tools, create_tool, tools as loaded_tools


@pytest.fixture
def sample_tools_json(tmp_path):
    tools = [
        {
            "name": "get_testEndpoint",
            "description": "Retrieves test data.",
            "endpoint": "/testEndpoint",
            "method": "GET",
        },
        {
            "name": "post_submitData",
            "description": "Submits data to test endpoint.",
            "endpoint": "/submitData",
            "method": "POST",
        },
    ]
    tools_json = tmp_path / "generated_tools.json"
    with open(tools_json, "w") as f:
        json.dump(tools, f, indent=4)
    return tools_json


def test_load_tools(sample_tools_json):
    with patch("tool_module.create_tool") as mock_create_tool:
        mock_tool_instance = MagicMock()
        mock_create_tool.return_value = mock_tool_instance
        loaded = load_tools(str(sample_tools_json))
        assert len(loaded) == 2
        mock_create_tool.assert_any_call(
            {
                "name": "get_testEndpoint",
                "description": "Retrieves test data.",
                "endpoint": "/testEndpoint",
                "method": "GET",
            }
        )
        mock_create_tool.assert_any_call(
            {
                "name": "post_submitData",
                "description": "Submits data to test endpoint.",
                "endpoint": "/submitData",
                "method": "POST",
            }
        )


def test_create_tool(sample_tools_json):
    tool_def = {
        "name": "get_testEndpoint",
        "description": "Retrieves test data.",
        "endpoint": "/testEndpoint",
        "method": "GET",
    }
    with patch("tool_module.requests.request") as mock_request:
        mock_response = MagicMock()
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json.return_value = {"key": "value"}
        mock_request.return_value = mock_response

        tool = create_tool(tool_def)
        assert tool.name == "get_testEndpoint"
        assert tool.description == "Retrieves test data."

        # Execute the tool function
        response = tool.func({"param1": "test"})
        mock_request.assert_called_once_with(
            method="GET",
            url="https://api.example.com/testEndpoint",
            headers={
                "Authorization": "Bearer YOUR_TOKEN",
                "Content-Type": "application/json",
            },
            params={"param1": "test"},
        )
        assert response == '{"key": "value"}'


def test_create_tool_with_non_json_response(sample_tools_json):
    tool_def = {
        "name": "get_testEndpoint",
        "description": "Retrieves test data.",
        "endpoint": "/testEndpoint",
        "method": "GET",
    }
    with patch("tool_module.requests.request") as mock_request:
        mock_response = MagicMock()
        mock_response.headers = {"Content-Type": "text/plain"}
        mock_response.text = "Plain text response"
        mock_request.return_value = mock_response

        tool = create_tool(tool_def)
        response = tool.func({"param1": "test"})
        assert response == "Plain text response"
