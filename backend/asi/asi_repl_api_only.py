import json
from typing import List, Dict
import requests
import os
from dotenv import load_dotenv

# Import ChatAI21 and PromptTemplate
from langchain_ai21 import ChatAI21
from langchain_core.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

# Get AI21 API key from environment variable
AI21_API_KEY = os.getenv("AI21_API_KEY")


def fetch_api_data(api_link: str, endpoints: List[Dict]) -> Dict:
    """
    Sends requests to the API endpoints and collects the outputs.

    Args:
        api_link (str): Base URL of the API.
        endpoints (List[Dict]): List of endpoint definitions.

    Returns:
        Dict: Collected outputs from the API endpoints and working request configurations.
    """
    data = {}
    working_requests = {}

    for endpoint in endpoints:
        name = endpoint.get("name")
        method = endpoint.get("method").upper()
        path = endpoint.get("path")
        parameters = endpoint.get("parameters", [])

        url = api_link + path

        # For the purpose of example, we'll use default parameters
        params = {param: "1" for param in parameters}

        try:
            response = requests.request(method=method, url=url, params=params)
            response.raise_for_status()
            data[name] = [response.json()]
            working_requests[name] = [{"method": method, "url": url, "params": params}]
        except requests.RequestException as e:
            print(f"Request failed for {name}: {str(e)}")
            data[name] = [{"error": f"{str(e)}"}]
            working_requests[name] = []

    return data, working_requests


def create_tools_from_api(
    api_context: str,
    api_link: str,
    documentation: Dict,
    context_data: Dict,
    working_requests: Dict,
) -> List[Dict]:
    """
    Creates LangChain tools based on the provided API context, link, documentation, context data, and working requests.
    Uses an LLM to generate the tool functions.

    Args:
        api_context (str): Description or context of the API.
        api_link (str): Base URL of the API.
        documentation (Dict): Documentation containing endpoint details.
        context_data (Dict): Context data from API outputs.
        working_requests (Dict): Working request configurations for each endpoint.

    Returns:
        List[Dict]: A list of tool configurations.
    """
    tools = []

    # Initialize the AI21 Chat model for code generation
    code_gen_model = ChatAI21(model="jamba-1.5-large", api_key=AI21_API_KEY)

    # Create a prompt template for code generation
    code_gen_template = """
    Create a Python function for a LangChain tool with the following specifications:

    Function name: {name}
    Description: {description}
    Parameters: {parameters}
    API URL: {url}
    HTTP Method: {method}
    Sample working params: {working_params}

    The function should use the requests library to make an API call and return the JSON response.
    Include error handling and any necessary imports. ONLY WRITE THE FUNCTION, DO NOT INCLUDE ANYTHING ELSE.

    Function:
    """
    code_gen_prompt = PromptTemplate.from_template(code_gen_template)

    # Open a file to write the generated code
    with open("generated_code.txt", "w") as code_file:
        for endpoint in documentation.get("endpoints", []):
            name = endpoint.get("name")
            description = endpoint.get("description")
            parameters = endpoint.get("parameters", [])

            # Include sample output from context_data
            sample_output = context_data.get(name, [{}])[
                0
            ]  # Get the first successful result

            # Get working request configuration
            working_request = working_requests.get(name, [])
            if working_request:
                working_request = working_request[0]  # Get the first working request
            else:
                # If no working request, use default values
                working_request = {
                    "method": endpoint.get("method", "GET"),
                    "url": api_link + endpoint.get("path", ""),
                    "params": {},
                }

            # Generate the tool function using the LLM
            code_gen_input = code_gen_prompt.format(
                name=name,
                description=description,
                parameters=parameters,
                url=working_request.get("url", api_link),
                method=working_request.get("method", "GET"),
                working_params=working_request.get("params", {}),
            )

            # Use invoke instead of predict
            response = code_gen_model.invoke([HumanMessage(content=code_gen_input)])
            generated_code = response.content

            # Remove any text before and including ```python and after and including the closing ```
            start_marker = "```python"
            end_marker = "```"
            start_index = generated_code.find(start_marker)
            if start_index != -1:
                generated_code = generated_code[start_index + len(start_marker) :]
            end_index = generated_code.rfind(end_marker)
            if end_index != -1:
                generated_code = generated_code[:end_index]
            generated_code = generated_code.strip()

            # Print the generated code for debugging
            print(f"\nGenerated code for {name}:")
            print(generated_code)

            # Write the generated code to the file
            code_file.write(f"\n\n--- Generated code for {name} ---\n")
            code_file.write(generated_code)

            # Create the tool configuration dictionary
            tool_config = {
                "name": name,
                "description": description,
                "function_code": generated_code,
                "parameters": parameters,
                "sample_output": sample_output,
            }
            tools.append(tool_config)

    print("\nGenerated code has been written to generated_code.txt")
    return tools


def write_tools_to_json(tools: List[Dict], filename: str):
    """
    Writes the tool configurations to a JSON file.

    Args:
        tools (List[Dict]): A list of tool configurations.
        filename (str): The filename for the JSON output.
    """
    with open(filename, "w") as f:
        json.dump(tools, f, indent=4)


# Example usage
if __name__ == "__main__":
    api_context = "Example API for JSONPlaceholder"
    api_link = "https://jsonplaceholder.typicode.com"
    documentation = {
        "endpoints": [
            {
                "name": "get_post",
                "method": "GET",
                "path": "/posts/1",
                "description": "Get a specific post",
                "parameters": [],
            },
            {
                "name": "get_comments",
                "method": "GET",
                "path": "/comments",
                "description": "Get comments for a post",
                "parameters": ["postId"],
            },
        ]
    }

    # Fetch data from the API endpoints and get working requests
    context_data, working_requests = fetch_api_data(
        api_link, documentation["endpoints"]
    )

    # Create tools from the API documentation, context data, and working requests
    tools = create_tools_from_api(
        api_context, api_link, documentation, context_data, working_requests
    )
    write_tools_to_json(tools, "tools.json")
    print("Tools have been written to tools.json")

    # Load tools from JSON and reconstruct functions
    try:
        with open("tools.json", "r") as f:
            tools_data = json.load(f)

        langchain_tools = []
        for tool_data in tools_data:
            # Add a check to ensure we're only executing Python code
            if tool_data["function_code"].strip().startswith("def "):
                exec(tool_data["function_code"], globals())
            else:
                print(f"Skipping invalid function code for tool: {tool_data['name']}")
            tool_function = globals()[tool_data["name"]]
            langchain_tool = Tool(
                name=tool_data["name"],
                func=tool_function,
                description=tool_data["description"],
            )
            langchain_tools.append(langchain_tool)

        # Initialize the AI21 Chat model for the agent
        agent_model = ChatAI21(model="jamba-1.5-large", api_key=AI21_API_KEY)

        # Create a prompt template
        template = """Question: {input}

Answer: Let's think step by step."""
        prompt = PromptTemplate.from_template(template)

        # Initialize the agent with the AI21 LLM and the tools
        agent = initialize_agent(
            tools=langchain_tools,
            llm=agent_model,
            agent="zero-shot-react-description",
            verbose=True,
        )

        # Now you can use the agent
        response = agent.run("What is the title of post 1?")
        print(response)
    except Exception as e:
        print(f"Error loading or executing tools: {e}")
