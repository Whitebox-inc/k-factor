# tests/test_integration.py

import pytest
from unittest.mock import patch, MagicMock
import json
import os

# Import modules
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
from kay.tool_module import load_tools
from kay.kay2 import KayAgent
from langchain.llms import AI21


@pytest.fixture
def sample_api_docs():
    return """
    Sample API Documentation:
    - Endpoint: /getData
      Method: GET
      Parameters: 
        - query (string): The search query.
        - limit (int): The number of results to return.
      Authentication: Bearer token
      Description: Retrieves data based on the query and limit.
    """


@pytest.fixture
def sample_endpoints():
    return ["/getData"]


def test_end_to_end_workflow(sample_api_docs, sample_endpoints, tmp_path):
    # Mock the LanguageModel's generate_completion
    with patch("asi_repl.AI21LLM") as MockLLM:
        mock_llm_instance = MockLLM.return_value
        # Define the response from the LLM
        mock_llm_instance.invoke.return_value = json.dumps(
            {
                "method": "GET",
                "headers": {"Authorization": "Bearer testtoken"},
                "params": {"query": "climate change", "limit": 5},
                "data": None,
                "description": "Retrieves climate change data.",
                "endpoint": "/getData",
            }
        )

        # Initialize the LanguageModel
        model = LanguageModel(api_key="test-api-key", model_name="j2-ultra")

        # Patch the current working directory to tmp_path
        with patch("asi_repl.open", new_callable=pytest.mock.mock_open()) as mock_file:
            # Run the graph
            tools, docs = run_graph_until_completion(
                api_docs=sample_api_docs,
                target_goal="200 OK",
                endpoints=sample_endpoints,
                model=model,
            )

            # Assert that tools were generated correctly
            assert len(tools) == 1
            assert tools[0]["name"] == "get_getData"
            assert tools[0]["method"] == "GET"
            assert tools[0]["description"] == "Retrieves climate change data."
            assert tools[0]["endpoint"] == "/getData"

            # Assert that documentation was generated
            expected_doc = (
                "# LlamaIndex Tools Documentation\n\n"
                "## get_getData\n"
                "**Endpoint:** `/getData`\n"
                "**Method:** `GET`\n"
                "**Description:** Retrieves climate change data.\n\n"
                "### Usage\n```python\nresponse = get_getData(params={'key': 'value'})\nprint(response)\n```\n\n"
            )
            assert docs == expected_doc


def test_kay_agent_integration(sample_api_docs, sample_endpoints, tmp_path):
    # Step 1: Generate tools using asi_repl.py
    with patch("asi_repl.AI21LLM") as MockLLM:
        mock_llm_instance = MockLLM.return_value
        mock_llm_instance.invoke.return_value = json.dumps(
            {
                "method": "GET",
                "headers": {"Authorization": "Bearer testtoken"},
                "params": {"query": "climate change", "limit": 5},
                "data": None,
                "description": "Retrieves climate change data.",
                "endpoint": "/getData",
            }
        )

        model = LanguageModel(api_key="test-api-key", model_name="j2-ultra")

        with patch("asi_repl.open", new_callable=pytest.mock.mock_open()):
            tools, docs = run_graph_until_completion(
                api_docs=sample_api_docs,
                target_goal="200 OK",
                endpoints=sample_endpoints,
                model=model,
            )

    # Step 2: Load tools into tool_module
    with patch("tool_module.requests.request") as mock_request:
        mock_response = MagicMock()
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json.return_value = {"status_code": 200, "data": {"key": "value"}}
        mock_request.return_value = mock_response

        loaded = load_tools("generated_tools.json")
        assert len(loaded) == 1
        tool = loaded[0]
        assert tool.name == "get_getData"

        # Step 3: Initialize KayAgent
        with patch("kay_agent.AI21") as MockLLM:
            mock_llm_instance = MockLLM.return_value
            mock_llm_instance.run.return_value = json.dumps(
                {"status_code": 200, "data": {"key": "value"}}
            )

            llm = AI21(model="j2-ultra", api_key="test-api-key")
            kay = KayAgent(llm=llm, tools=loaded)

            # Step 4: Handle user input
            user_input = "Retrieve climate change data with a limit of 5 results."
            response = kay.handle_user_input(user_input)

            # Verify the response
            expected_response = '{"status_code": 200, "data": {"key": "value"}}\n\n'
            assert response == expected_response
