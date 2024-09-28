import json
import logging
from langchain.agents import initialize_agent, AgentType
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.llms import AI21
from tool_module import tools  # Import the tools loaded from tool_module.py
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KayAgent:
    """LangChain agent named Kay that utilizes generated tools."""

    def __init__(self, llm: AI21, tools: list):
        self.llm = llm
        self.tools = tools
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
        )

    def handle_user_input(self, user_input: str) -> str:
        """Processes user input, executes necessary tools, and returns the response."""
        try:
            logger.info(f"User input: {user_input}")
            response = self.agent.run(user_input)
            logger.info(f"Agent response: {response}")
            # Optionally, process the response for visualizations
            visualized_response = self.process_response(response)
            return visualized_response
        except Exception as e:
            logger.error(f"Error handling user input: {e}")
            return f"Error: {str(e)}"

    def process_response(self, response: str) -> str:
        """Processes the response to include visualizations if applicable."""
        try:
            # Attempt to parse the response as JSON
            data = json.loads(response)
            if isinstance(data, dict):
                # Example: Create a bar chart if data contains numerical values
                keys = list(data.keys())
                values = list(data.values())
                if all(isinstance(v, (int, float)) for v in values):
                    plt.figure(figsize=(10, 6))
                    plt.bar(keys, values, color="skyblue")
                    plt.xlabel("Keys")
                    plt.ylabel("Values")
                    plt.title("Visualization of API Response Data")
                    plt.tight_layout()

                    # Save plot to a bytes buffer
                    buf = BytesIO()
                    plt.savefig(buf, format="png")
                    plt.close()
                    buf.seek(0)
                    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
                    # Embed image in HTML
                    img_html = f'<img src="data:image/png;base64,{img_base64}" alt="Visualization"/>'
                    return f"{json.dumps(data, indent=2)}\n\n{img_html}"
            # If not applicable for visualization, return the raw response
            return response
        except json.JSONDecodeError:
            # If response is not JSON, return as is
            return response
        except Exception as e:
            logger.error(f"Error processing response for visualization: {e}")
            return response

    def display_actions(self):
        """Displays the conversation history."""
        return self.memory.buffer


# Example usage
if __name__ == "__main__":
    # Initialize the LLM (JAMBA via LangChain)
    llm = AI21(model="j2-ultra", api_key="your-jamba-api-key")

    # Initialize Kay with the loaded tools
    kay = KayAgent(llm=llm, tools=tools)

    # Example user input
    user_input = (
        "I want to retrieve data about climate change with a limit of 5 results."
    )
    response = kay.handle_user_input(user_input)
    print("Kay's Response:")
    print(response)

    # Display conversation history
    history = kay.display_actions()
    print("\nConversation History:")
    print(history)
