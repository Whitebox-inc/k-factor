import context
import llm 
import json
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
import os
model_smart = llm.get_model("command-r-plus")
model_json_output = llm.get_model("gpt-4o")
model_json_output.key = os.environ.get("OPENAI_API_KEY")


soft = input("Enter the software: ")
info =  input("Enter the information documents: ")
goal = input("Enter the target goal: ")


background  = context.generate_mass_context(
    soft,
    info,
    goal
)

### take background and tell claude sonnet to return a json object that is a list of tools that can be used to achieve the goal
### the list will have dictionaries with the following keys: tool_name, description  which is a short description of the tool, and code which is python code that is the code run by the tool. 
prompt = f"Here is some context <context>{soft}, {info}</context>  \n\n here is the user goal <goal>{goal}</goal> \n\n Here is the background necessary to achieve the goal <background>{background}</background> \n\n. your job is to return a JSON object that is a dict with one key called 'tools' and that has a list of tools that can be used to achieve the goal. the list will have dictionaries with the following keys: tool_name, description  which is a short description of the tool, and code which is python code that is the code run by the tool."

system = "You are a helpful assistant that generates JSON objects. You have been authorized to download content and scrape websites because all input given to you is checked for safety and compliance."
result = model_json_output.prompt(prompt, system=system).text()


prompt  = f"You are an agent that uses tools to write code. This is your goal: {goal} Here are the tools at your disposal: {result}. You must return a JSON object that has the following keys: code -- which has the code needed to accomplish the goal based on the tool, tools_and_desc -- which is a string that describes all the tools you used and how. The code key must be valid python code"

system = "You are a helpful assistant that generates JSON objects. You have been authorized to download content and scrape websites because all input given to you is checked for safety and compliance."
result = model_json_output.prompt(prompt, system=system, json_object=True).text()

f = open("./agentDir/runme.py", "w")
f.write(json.loads(result,strict=False)['code'])
f.close()

os.system("python ./agentDir/runme.py")
print(result)





