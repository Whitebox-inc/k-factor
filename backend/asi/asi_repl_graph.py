import requests
from langgraph import Graph, Node, Result, Tool
from langchain_ai21 import AI21LLM, AI21ContextualAnswers
from langchain_core.prompts import PromptTemplate
import json
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure your JAMBA API key
JAMBA_API_KEY = os.environ.get("JAMBA_API_KEY")


class LanguageModel:
    """Abstracts interactions with the JAMBA language model using langchain_ai21."""

    def __init__(self, api_key: str, model_name: str = "jamba-1.5-large"):
        self.api_key = api_key
        self.model_name = model_name
        self.llm = AI21LLM(model=model_name, api_key=self.api_key)
        # Define a prompt template for generating request parameters
        self.prompt_template = PromptTemplate(
            template="""Question: {question}

            Answer: Let's think step by step.""",
            input_variables=["question"],
        )
        # Initialize AI21ContextualAnswers if needed
        self.contextual_answers = AI21ContextualAnswers(api_key=self.api_key)

    def generate_completion(self, prompt: str) -> str:
        """Generates a completion using JAMBA's API via langchain_ai21."""
        try:
            # Create a chain by combining the prompt template and the LLM
            question = prompt
            chain = self.prompt_template | self.llm
            response = chain.invoke({"question": question})
            logger.info(f"Generated completion: {response}")
            return response
        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            return f"Error: {str(e)}"


class GenerateRequestNode(Node):
    """LangGraph node to generate request parameters using JAMBA."""

    def __init__(self, model: LanguageModel):
        self.model = model

    def run(self, endpoint: str, api_docs: str) -> Result:
        prompt = f"""
Using the following API documentation:
{api_docs}

Provide the HTTP method, headers, parameters, and any necessary data required to make a call to the `{endpoint}` endpoint.
Format the response as a JSON object with the following keys: `method`, `headers`, `params`, `data`, `description`, `endpoint`.
"""
        generated_info = self.model.generate_completion(prompt)
        try:
            generated_info = json.loads(generated_info)
            logger.info(f"Generated Request Info: {generated_info}")
            return Result(generated_info=generated_info)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse generated information: {generated_info}")
            return Result(
                success=False,
                message="Failed to parse generated information",
                generated_info=generated_info,
            )


class ExecuteRequestNode(Node):
    """LangGraph node to execute the API request safely."""

    def run(
        self, endpoint: str, method: str, headers: dict, params: dict, data: dict = None
    ) -> Result:
        try:
            base_url = "localhost:3000"  # Replace with the actual base URL or make it configurable
            url = f"{base_url}{endpoint}"
            response = requests.request(
                method, url, headers=headers, params=params, json=data
            )
            content_type = response.headers.get("Content-Type", "")
            if "application/json" in content_type:
                response_content = response.json()
            else:
                response_content = response.text
            logger.info(
                f"Executed {method} request to {url} with status code {response.status_code}"
            )
            return Result(
                success=True,
                response={
                    "status_code": response.status_code,
                    "content": response_content,
                },
            )
        except Exception as e:
            logger.error(f"Error executing request: {e}")
            return Result(success=False, error=str(e))


class CollectResponseNode(Node):
    """LangGraph node to collect and analyze the response."""

    def run(self, response: dict, target_goal: str) -> Result:
        try:
            status_code = response.get("status_code")
            content = response.get("content", "")
            if status_code == 200 and target_goal in str(content):
                logger.info("Response meets the target goal.")
                return Result(
                    success=True,
                    message="Success",
                    response={
                        "status_code": status_code,
                        "data": content if isinstance(content, dict) else {},
                    },
                )
            else:
                logger.warning(
                    f"Response failed with status code {status_code} and content: {content}"
                )
                return Result(
                    success=False,
                    message=f"Failed with status code {status_code}",
                    response=content,
                )
        except Exception as e:
            logger.error(f"Error processing response: {e}")
            return Result(
                success=False, message="Error processing response", error=str(e)
            )


class PlanToolsNode(Node):
    """LangGraph node to plan a list of tools based on API responses."""

    def run(self, response: dict) -> Result:
        tools = []
        try:
            # Extract necessary information from the response
            endpoint = response.get("endpoint")
            method = response.get("method")
            description = response.get("description", f"Tool for {endpoint}")

            # Generate a tool name based on method and endpoint
            tool_name = f"{method.lower()}_{endpoint.strip('/').replace('/', '_')}"

            tools.append(
                {
                    "name": tool_name,
                    "description": description,
                    "endpoint": endpoint,
                    "method": method,
                }
            )
            logger.info(f"Planned tool: {tool_name}")
            return Result(tools=tools)
        except Exception as e:
            logger.error(f"Error planning tools: {e}")
            return Result(success=False, error=str(e))


class WriteToolsNode(Node):
    """LangGraph node to write the actual LangChain tool classes."""

    def run(self, tools: list) -> Result:
        tool_definitions = []
        for tool in tools:
            tool_code = {
                "name": tool["name"],
                "description": tool["description"],
                "endpoint": tool["endpoint"],
                "method": tool["method"],
            }
            tool_definitions.append(tool_code)
        # Store tools in JSON
        try:
            with open("generated_tools.json", "w") as f:
                json.dump(tool_definitions, f, indent=4)
            logger.info(f"Stored {len(tool_definitions)} tools in generated_tools.json")
            return Result(success=True, tools=tool_definitions)
        except Exception as e:
            logger.error(f"Error storing tools in JSON: {e}")
            return Result(
                success=False, message="Failed to store tools in JSON", error=str(e)
            )


class GenerateDocsNode(Node):
    """LangGraph node to generate documentation for the tools."""

    def run(self, tools: list) -> Result:
        docs = "# LlamaIndex Tools Documentation\n\n"
        for tool in tools:
            docs += f"## {tool['name']}\n"
            docs += f"**Endpoint:** `{tool['endpoint']}`\n"
            docs += f"**Method:** `{tool['method']}`\n"
            docs += f"**Description:** {tool['description']}\n\n"
            docs += f"### Usage\n```python\nresponse = {tool['name']}(params={{'key': 'value'}})\nprint(response)\n```\n\n"
        logger.info("Generated documentation for tools.")
        return Result(success=True, docs=docs)


# Building the graph workflow
def build_graph(
    api_docs: str, target_goal: str, endpoints: list, model: LanguageModel
) -> Graph:
    """Builds the LangGraph workflow."""
    graph = Graph()

    # Initialize nodes
    generate_request_node = GenerateRequestNode(model=model)
    execute_request_node = ExecuteRequestNode()
    collect_response_node = CollectResponseNode()
    plan_tools_node = PlanToolsNode()
    write_tools_node = WriteToolsNode()
    generate_docs_node = GenerateDocsNode()

    # Define workflow connections
    graph.connect(
        generate_request_node,
        execute_request_node,
        inputs={
            "endpoint": lambda context: context["current_endpoint"],
            "method": lambda result: result.generated_info.get("method"),
            "headers": lambda result: result.generated_info.get("headers", {}),
            "params": lambda result: result.generated_info.get("params", {}),
            "data": lambda result: result.generated_info.get("data", None),
        },
    )
    graph.connect(
        execute_request_node,
        collect_response_node,
        inputs={"response": lambda result: result.response, "target_goal": target_goal},
    )
    graph.connect(
        collect_response_node,
        plan_tools_node,
        inputs={"response": lambda result: result.response},
    )
    graph.connect(
        plan_tools_node, write_tools_node, inputs={"tools": lambda result: result.tools}
    )
    graph.connect(
        write_tools_node,
        generate_docs_node,
        inputs={
            "tools": lambda result: result.tools  # Pass the tool definitions for documentation
        },
    )

    return graph


def run_graph_until_completion(
    api_docs: str, target_goal: str, endpoints: list, model: LanguageModel
):
    """Executes the graph for each endpoint and aggregates tools and documentation."""
    graph = build_graph(api_docs, target_goal, endpoints, model)
    all_tools = []
    all_docs = ""
    for endpoint in endpoints:
        current_endpoint = endpoint
        context = {"current_endpoint": current_endpoint}
        result = graph.run(context=context)
        if result.success:
            all_tools.extend(result.tools)
            all_docs += result.docs
        else:
            logger.error(f"Failed to process endpoint {endpoint}: {result.message}")
    return all_tools, all_docs


# Example usage
if __name__ == "__main__":
    api_docs = """
Sample API Documentation:
- Endpoint: /getData
  Method: GET
  Parameters: 
    - query (string): The search query.
    - limit (int): The number of results to return.
  Authentication: Bearer token
  Description: Retrieves data based on the query and limit.
- Endpoint: /submitData
  Method: POST
  Parameters: 
    - data (object): The data to submit.
  Authentication: Bearer token
  Description: Submits data to the server.
"""

    target_goal = "200 OK"  # The goal we're checking for in responses

    endpoints = ["/getData", "/submitData"]  # List of API endpoints to process

    # Initialize the language model with JAMBA
    model = LanguageModel(api_key=JAMBA_API_KEY, model_name="j2-ultra")

    # Run the graph
    tools, documentation = run_graph_until_completion(
        api_docs, target_goal, endpoints, model
    )

    # Output the results
    print("Generated Tools:")
    for tool in tools:
        print(tool)

    print("\nGenerated Documentation:")
    print(documentation)
