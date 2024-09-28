import json
import os
from typing import List, Dict
from dotenv import load_dotenv
from langchain_ai21 import ChatAI21
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Get AI21 API key from environment variable
AI21_API_KEY = os.getenv("AI21_API_KEY")


class Kay3Agent:
    """LangChain agent named Kay3 that utilizes tools generated by asi_repl_api_only.py"""

    def __init__(self, tools_file: str):
        self.llm = ChatAI21(model="jamba-1.5-large", api_key=AI21_API_KEY)
        self.tools = self.load_tools(tools_file)
        print(f"Loaded {len(self.tools)} tools")
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        self.agent = self.initialize_agent()

    def load_tools(self, tools_file: str) -> List[Tool]:
        """Load tools from the JSON file and create LangChain Tool objects"""
        with open(tools_file, "r") as f:
            tools_data = json.load(f)

        langchain_tools = []
        for tool_data in tools_data:
            try:
                # Create a new function using exec
                exec(tool_data["function_code"], globals())
                tool_function = globals()[tool_data["name"]]

                langchain_tool = Tool(
                    name=tool_data["name"],
                    func=tool_function,
                    description=tool_data["description"],
                )
                langchain_tools.append(langchain_tool)
                print(f"Successfully loaded tool: {tool_data['name']}")
            except Exception as e:
                print(f"Error loading tool {tool_data['name']}: {str(e)}")
                print(f"Function code:\n{tool_data['function_code']}")

        return langchain_tools

    def initialize_agent(self) -> AgentType:
        """Initialize the LangChain agent with loaded tools and AI21 LLM"""
        return initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            handle_parsing_errors=True,
        )

    def handle_user_input(self, user_input: str) -> str:
        """Process user input and return the agent's response"""
        try:
            response = self.agent.run(input=user_input)
            return response
        except Exception as e:
            return f"An error occurred: {str(e)}"

    def display_conversation_history(self) -> str:
        """Return the conversation history as a formatted string"""
        history = self.memory.chat_memory.messages
        formatted_history = ""
        for message in history:
            if message.type == "human":
                formatted_history += f"Human: {message.content}\n"
            elif message.type == "ai":
                formatted_history += f"AI: {message.content}\n"
        return formatted_history


# Example usage
if __name__ == "__main__":
    # Initialize Kay3 with the tools generated by asi_repl_api_only.py
    kay3 = Kay3Agent("tools.json")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Kay3: Goodbye!")
            break

        response = kay3.handle_user_input(user_input)
        print(f"Kay3: {response}")

        print("\nConversation History:")
        print(kay3.display_conversation_history())