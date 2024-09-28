import json
from langchain.agents import Tool
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_tools(json_file: str) -> list:
    """Loads tool definitions from a JSON file and creates LangChain Tool instances."""
    tools = []
    try:
        with open(json_file, "r") as f:
            tool_definitions = json.load(f)
        for tool_def in tool_definitions:
            tool = create_tool(tool_def)
            tools.append(tool)
        logger.info(f"Loaded {len(tools)} tools from {json_file}")
    except Exception as e:
        logger.error(f"Error loading tools from {json_file}: {e}")
    return tools


def create_tool(tool_def: dict) -> Tool:
    """Creates a LangChain Tool instance from a tool definition."""

    def tool_func(params: dict) -> str:
        """Dynamically generated tool function."""
        url = f"https://api.example.com{tool_def['endpoint']}"
        headers = {
            "Authorization": "Bearer YOUR_TOKEN",  # Replace with actual token or implement secure retrieval
            "Content-Type": "application/json",
        }
        try:
            response = requests.request(
                method=tool_def["method"], url=url, headers=headers, params=params
            )
            if "application/json" in response.headers.get("Content-Type", ""):
                result = response.json()
            else:
                result = response.text
            return str(result)
        except Exception as e:
            logger.error(f"Error executing tool {tool_def['name']}: {e}")
            return f"Error: {str(e)}"

    return Tool(
        name=tool_def["name"], func=tool_func, description=tool_def["description"]
    )


# Load tools on module import
tools = load_tools("generated_tools.json")
