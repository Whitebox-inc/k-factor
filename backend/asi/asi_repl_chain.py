import requests
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_ai21 import AI21LLM
from langchain.chains import SequentialChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json
import logging
import os
from langchain.chains.base import Chain
from langchain.llms.base import BaseLLM

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

    def get_llm(self):
        return self.llm


class RequestInfo(BaseModel):
    method: str = Field(description="HTTP method for the request")
    headers: dict = Field(description="Headers for the request")
    params: Optional[dict] = Field(description="Query parameters for the request")
    data: Optional[dict] = Field(description="Data payload for the request")
    description: str = Field(description="Description of the request")
    endpoint: str = Field(description="API endpoint")


class GenerateRequestChain(Chain):
    llm: BaseLLM
    prompt: PromptTemplate
    output_key: str = "request_info"

    def __init__(self, llm: BaseLLM, **kwargs):
        prompt = PromptTemplate(
            template="""
            Using the following API documentation:
            {api_docs}

            Provide the HTTP method, headers, parameters, and any necessary data required to make a call to the `{endpoint}` endpoint.
            {format_instructions}
            """,
            input_variables=["api_docs", "endpoint"],
            partial_variables={
                "format_instructions": json.dumps(RequestInfo.model_json_schema())
            },
        )
        super().__init__(llm=llm, prompt=prompt, **kwargs)

    @property
    def input_keys(self) -> List[str]:
        return self.prompt.input_variables

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        prompt_value = self.prompt.format(**inputs)
        response = self.llm(prompt_value)
        return {self.output_key: response}

    @property
    def _chain_type(self) -> str:
        return "generate_request_chain"


class ExecuteRequestChain:
    def __init__(self, base_url="localhost:3000"):
        self.base_url = base_url

    def run(self, request_info: RequestInfo):
        try:
            url = f"{self.base_url}{request_info.endpoint}"
            response = requests.request(
                request_info.method,
                url,
                headers=request_info.headers,
                params=request_info.params,
                json=request_info.data,
            )
            content_type = response.headers.get("Content-Type", "")
            if "application/json" in content_type:
                response_content = response.json()
            else:
                response_content = response.text
            logger.info(
                f"Executed {request_info.method} request to {url} with status code {response.status_code}"
            )
            return {
                "status_code": response.status_code,
                "content": response_content,
            }
        except Exception as e:
            logger.error(f"Error executing request: {e}")
            return {"error": str(e)}


class CollectResponseChain:
    def run(self, response: dict, target_goal: str):
        try:
            status_code = response.get("status_code")
            content = response.get("content", "")
            if status_code == 200 and target_goal in str(content):
                logger.info("Response meets the target goal.")
                return {
                    "success": True,
                    "message": "Success",
                    "response": {
                        "status_code": status_code,
                        "data": content if isinstance(content, dict) else {},
                    },
                }
            else:
                logger.warning(
                    f"Response failed with status code {status_code} and content: {content}"
                )
                return {
                    "success": False,
                    "message": f"Failed with status code {status_code}",
                    "response": content,
                }
        except Exception as e:
            logger.error(f"Error processing response: {e}")
            return {
                "success": False,
                "message": "Error processing response",
                "error": str(e),
            }


class PlanToolsChain:
    def run(self, response: dict):
        tools = []
        try:
            endpoint = response.get("endpoint")
            method = response.get("method")
            description = response.get("description", f"Tool for {endpoint}")

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
            return {"tools": tools}
        except Exception as e:
            logger.error(f"Error planning tools: {e}")
            return {"error": str(e)}


class WriteToolsChain:
    def run(self, tools: List[dict]):
        try:
            with open("generated_tools.json", "w") as f:
                json.dump(tools, f, indent=4)
            logger.info(f"Stored {len(tools)} tools in generated_tools.json")
            return {"success": True, "tools": tools}
        except Exception as e:
            logger.error(f"Error storing tools in JSON: {e}")
            return {
                "success": False,
                "message": "Failed to store tools in JSON",
                "error": str(e),
            }


class GenerateDocsChain:
    def run(self, tools: List[dict]):
        docs = "# LlamaIndex Tools Documentation\n\n"
        for tool in tools:
            docs += f"## {tool['name']}\n"
            docs += f"**Endpoint:** `{tool['endpoint']}`\n"
            docs += f"**Method:** `{tool['method']}`\n"
            docs += f"**Description:** {tool['description']}\n\n"
            docs += f"### Usage\n```python\nresponse = {tool['name']}(params={{'key': 'value'}})\nprint(response)\n```\n\n"
        logger.info("Generated documentation for tools.")
        return {"success": True, "docs": docs}


def build_chain(model: LanguageModel):
    """Builds the LangChain workflow."""
    llm = model.get_llm()

    generate_request_chain = GenerateRequestChain(llm)
    execute_request_chain = ExecuteRequestChain()
    collect_response_chain = CollectResponseChain()
    plan_tools_chain = PlanToolsChain()
    write_tools_chain = WriteToolsChain()
    generate_docs_chain = GenerateDocsChain()

    # Create a SequentialChain to combine all the steps
    chain = SequentialChain(
        chains=[
            generate_request_chain,
            execute_request_chain,
            collect_response_chain,
            plan_tools_chain,
            write_tools_chain,
            generate_docs_chain,
        ],
        input_variables=["api_docs", "endpoint", "target_goal"],
        output_variables=["tools", "docs"],
        verbose=True,
    )

    return chain


def run_chain_until_completion(
    api_docs: str, target_goal: str, endpoints: list, model: LanguageModel
):
    """Executes the chain for each endpoint and aggregates tools and documentation."""
    chain = build_chain(model)
    all_tools = []
    all_docs = ""
    for endpoint in endpoints:
        result = chain.run(
            api_docs=api_docs, endpoint=endpoint, target_goal=target_goal
        )
        all_tools.extend(result.get("tools", []))
        all_docs += result.get("docs", "")
    return all_tools, all_docs


if __name__ == "__main__":
    model = LanguageModel(JAMBA_API_KEY)
    chain = build_chain(model)
    api_docs = "API documentation is not provided"
    target_goal = "Target goal"
    endpoints = ["/endpoint1", "/endpoint2"]
    all_tools, all_docs = run_chain_until_completion(
        api_docs, target_goal, endpoints, model
    )
    print("All tools:", all_tools)
    print("All docs:", all_docs)

    # Example case using jsonplaceholder.typicode.com
    print("\nExample case using jsonplaceholder.typicode.com:")

    jsonplaceholder_api_docs = """
    JSONPlaceholder is a free online REST API that you can use whenever you need some fake data.
    It's great for tutorials, testing new libraries, sharing code examples, ...

    Resources:
    - /posts	100 posts
    - /comments	500 comments
    - /albums	100 albums
    - /photos	5000 photos
    - /todos	200 todos
    - /users	10 users

    Routes:
    GET	/posts
    GET	/posts/1
    GET	/posts/1/comments
    GET	/comments?postId=1
    POST	/posts
    PUT	/posts/1
    PATCH	/posts/1
    DELETE	/posts/1
    """

    jsonplaceholder_model = LanguageModel(JAMBA_API_KEY)
    jsonplaceholder_chain = build_chain(jsonplaceholder_model)

    # Override the base_url in ExecuteRequestChain
    for chain in jsonplaceholder_chain.chains:
        if isinstance(chain, ExecuteRequestChain):
            chain.base_url = "https://jsonplaceholder.typicode.com"

    result = jsonplaceholder_chain.run(
        api_docs=jsonplaceholder_api_docs, endpoint="/posts/1", target_goal="title"
    )

    print("JSONPlaceholder example result:")
    print(json.dumps(result, indent=2))
