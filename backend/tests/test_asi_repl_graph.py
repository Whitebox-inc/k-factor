# tests/test_asi_repl.py

import pytest
from unittest.mock import patch, MagicMock
import json
import os

# Import classes from asi_repl.py
from asi.asi_repl_graph import (
    LanguageModel,
    GenerateRequestNode,
    ExecuteRequestNode,
    CollectResponseNode,
    PlanToolsNode,
    WriteToolsNode,
    GenerateDocsNode,
    build_graph,
    run_graph_until_completion,
)


@pytest.fixture
def mock_language_model():
    with patch("asi_repl.AI21LLM") as MockLLM:
        mock_llm_instance = MockLLM.return_value
        mock_llm_instance.invoke.return_value = '{"method": "GET", "headers": {"Authorization": "Bearer testtoken"}, "params": {"query": "test", "limit": 5}, "data": null, "description": "Test description", "endpoint": "/testEndpoint"}'
        model = LanguageModel(api_key="test-api-key", model_name="j2-ultra")
        yield model


def test_generate_request_node(mock_language_model):
    node = GenerateRequestNode(model=mock_language_model)
    result = node.run(endpoint="/testEndpoint", api_docs="Test API Documentation")
    assert result.success
    assert result.generated_info["method"] == "GET"
    assert result.generated_info["headers"]["Authorization"] == "Bearer testtoken"
    assert result.generated_info["params"]["query"] == "test"
    assert result.generated_info["params"]["limit"] == 5
    assert result.generated_info["data"] is None
    assert result.generated_info["description"] == "Test description"
    assert result.generated_info["endpoint"] == "/testEndpoint"


def test_write_tools_node(tmp_path):
    # Prepare sample tools
    tools = [
        {
            "name": "get_testEndpoint",
            "description": "Test description",
            "endpoint": "/testEndpoint",
            "method": "GET",
        }
    ]

    # Initialize WriteToolsNode
    node = WriteToolsNode()
    # Change current working directory to tmp_path to avoid file system side effects
    with patch("builtins.open", new_callable=pytest.mock.mock_open()) as mock_file:
        result = node.run(tools)
        mock_file.assert_called_once_with("generated_tools.json", "w")
        handle = mock_file()
        handle.write.assert_called_once_with(json.dumps(tools, indent=4))
        assert result.success
        assert result.tools == tools


def test_generate_docs_node():
    node = GenerateDocsNode()
    tools = [
        {
            "name": "get_testEndpoint",
            "description": "Test description",
            "endpoint": "/testEndpoint",
            "method": "GET",
        }
    ]
    result = node.run(tools)
    assert result.success
    expected_doc = (
        "# LlamaIndex Tools Documentation\n\n"
        "## get_testEndpoint\n"
        "**Endpoint:** `/testEndpoint`\n"
        "**Method:** `GET`\n"
        "**Description:** Test description\n\n"
        "### Usage\n```python\nresponse = get_testEndpoint(params={'key': 'value'})\nprint(response)\n```\n\n"
    )
    assert result.docs == expected_doc


def test_run_graph_until_completion(mock_language_model, tmp_path):
    # Create a temporary generated_tools.json
    sample_tools = [
        {
            "name": "get_testEndpoint",
            "description": "Test description",
            "endpoint": "/testEndpoint",
            "method": "GET",
        }
    ]
    tools_json = tmp_path / "generated_tools.json"
    with open(tools_json, "w") as f:
        json.dump(sample_tools, f)

    # Mock run_graph_until_completion to use the temporary generated_tools.json
    with patch("asi_repl.build_graph") as mock_build_graph:
        mock_graph = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.tools = sample_tools
        mock_result.docs = "Generated documentation"
        mock_graph.run.return_value = mock_result
        mock_build_graph.return_value = mock_graph

        # Run the function
        tools, docs = run_graph_until_completion(
            api_docs="Test API Documentation",
            target_goal="200 OK",
            endpoints=["/testEndpoint"],
            model=mock_language_model,
        )

        # Assertions
        mock_build_graph.assert_called_once()
        mock_graph.run.assert_called_once_with(
            context={"current_endpoint": "/testEndpoint"}
        )
        assert tools == sample_tools
        assert docs == "Generated documentation"
