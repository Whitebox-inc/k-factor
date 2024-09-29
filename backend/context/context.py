from ai21 import AI21Client
from ai21.models.chat import ChatMessage
import re 
import json
from firecrawl import FirecrawlApp
from tqdm import tqdm
import llm 

sonnet = llm.get_model("claude-3.5-sonnet")

app = FirecrawlApp(api_key=json.load(open("keys.json"))["firecrawl"])



from openperplex import Openperplex

api_key = json.load(open("keys.json"))["openperplex"]
opplx_client = Openperplex(api_key)


def scrape_website(url):
    try:
        scrape_result = app.scrape_url(url, params={'formats': ['markdown']})
        print("SCRAPE RESULT: ", scrape_result)
        if "markdown" in scrape_result:
            return scrape_result["markdown"]
        else:
            return False
    except Exception as e:
        return False

client = AI21Client(
    # defaults to os.enviorn.get('AI21_API_KEY')
    api_key=json.load(open("keys.json"))["ai21"],
)

system = "You're a support engineer in a SaaS company"
messages = [
    ChatMessage(content=system, role="system"),
    ChatMessage(content="Hello, I need help with a signup process.", role="user"),
]

chat_completions = client.chat.completions.create(
    messages=messages,
    model="jamba-1.5-mini",
)


def format_enforcer(agent, prompt, format_regex, num_tries=3, bRule="YOU MUST RETURN AN OUTPUT FOLLOWING THE SPECIFIED FORMAT", bRuleThreshold=1, lenient=True):
    result = agent(prompt)
    if lenient:
        result = result.strip()
    if not re.match(format_regex, result):
        if num_tries == 0:
            return None
        if num_tries == bRuleThreshold:
            prompt = prompt + "\n" + bRule
        return format_enforcer(agent, prompt, format_regex, num_tries-1, bRule, bRuleThreshold)
    return result


def basic_agent(prompt):
    chat_completions = client.chat.completions.create(
        messages=[
            ChatMessage(content="You are a rational Assistant, you answer questions and follow instructions.", role="system"),
            ChatMessage(content=prompt, role="user"),
        ],
        model="jamba-1.5-large",
    )
    return chat_completions.choices[0].message.content



def is_info_agent(infotext):
    prompt = "The user was given the following prompt: 'Enter documentation if you have any.'. The user has entered the following information: " + infotext + ". Has the user provided any information about the software? ANSWER ONLY WITH YES ALL CAPS."
    response = "YES" in basic_agent(prompt).upper()
    print("RESPONSE: ", response)
    if response:
        return True
    else:
        return not False
    
def extract_info_agent(infotext):

    #url regex = [-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)
    is_info = is_info_agent(infotext)
    print("IS INFO: ", is_info)
    info_object = {
        "url": None,
        "semantical_description": None,
    }
    if is_info:
        does_this_contain_url = basic_agent("Does the following text contain a URL, please answer YES or NO: " + infotext)
        if "YES" in does_this_contain_url.upper():
            url = basic_agent("Extract the URL from the following text. GIVE ME JUST THE URL NOTHING ELSE: " + infotext).strip()
            if url[-1] == ".":
                url = url[:-1]
            info_object["url"] = url
            info_object["semantical_description"] = infotext.replace(url, "").strip()
            return info_object
        else:
            info_object["semantical_description"] = infotext
            return info_object
    else:
        return None

    


def question_agent(target_goal, software):
    #this agent takes in the information about the software, and generates questions that another agent will answer. 
    #the answers will be used to generate a guide/context on how to use the software. 
    
    #first lets extract the information from the infotext
   
    
    
    prompt = f"The user has the following goal: <user_goal>{target_goal}</user_goal>. The user has the following software: <user_software>{software}</user_software>. Generate a list of questions about the software whose answers could help a developer trying to use the software to achieve the goal. Return a JSON object with the following structure: {{'questions':['question1','question2','question3'...]}}. Some example questions are: what is the software(is it a website, api, cli, etc), what protocol can I use to interface with it?,  and more. Please ask only useful questions, and write them as if you were putting them in a search engine (meaning make them specific):"

    system = "You're a support engineer in a SaaS company"
    messages = [
        ChatMessage(content="You are a helpful assistant that follows instructions and generates JSON objects.", role="system"),
        ChatMessage(content=prompt, role="user"),
    ]

    chat_completions = client.chat.completions.create(
        messages=messages,
        model="jamba-1.5-large",
    )

    response = chat_completions.choices[0].message.content
    try:
        response_json = json.loads(response)
        return response_json["questions"]
    except :
        return None





def answer_agent_from_internet(question, software, target_goal):
    #search the question using openperplex
    result = opplx_client.search(
        query=question,
        location="us", # default is "us"
        pro_mode=False, # default is False
        response_language="en", # default is "auto",
        answer_type="text", # options : 'html','markdown'
        search_type="general", # options : 'general' or 'news'
        verbose_mode= False# you can set it to True for long format responses
    )
    return result["llm_response"]

def answer_agent_from_context(question, software, target_goal, context):
    prompt = f"The user has the following goal: <user_goal>{target_goal}</user_goal>. The user has the following software: <user_software>{software}</user_software>. Here is some useful context about the software: <user_context>{context}</user_context>. Generate an answer to the following question about how the user can use the software to achieve their goal: {question}. If there is no information about the software, please answer with 'I'm sorry, I don't have any information about that.'"

    system = "You're a support engineer in a SaaS company"
    messages = [
        ChatMessage(content="You are a helpful assistant that follows instructions and answers questions well", role="system"),
        ChatMessage(content=prompt, role="user"),
    ]

    chat_completions = client.chat.completions.create(
        messages=messages,
        model="jamba-1.5-large",
    )

    response = chat_completions.choices[0].message.content
    return response


def discriminate_questions(questions, software, target_goal):
    #an agent that looks at all the questions and returns the most useful ones
    prompt = f"The user has the following goal: <user_goal>{target_goal}</user_goal>. The user has the following software: <user_software>{software}</user_software>. Here is a list of questions that were generated: {questions}. Return a list of the most useful questions for a developer to use the software to achieve their goal. Pay particular attention to questions that are about how to use software and code to interact with the software. Contrived questions about safety and security are not useful. Return a JSON object with the following structure: {{'questions':['question1','question2','question3'...]}}."

    system = "You're a support engineer in a SaaS company"
    messages = [
        ChatMessage(content="You are a helpful assistant that follows instructions and generates JSON objects.", role="system"),
        ChatMessage(content=prompt, role="user"),
    ]

    chat_completions = client.chat.completions.create(
        messages=messages,
        model="jamba-1.5-large",
    )

    response = chat_completions.choices[0].message.content
    try:
        response_json = json.loads(response)
        return response_json["questions"]
    except:
        return None

def generate_mass_context(software, info_docs, target_goal):
    # software = input("Enter the software you want to use: ")
    # info_docs = input("Enter documentation if you have any: ")
    # target_goal = input("Enter your desired output: text, code, agent, etc:")


    info = extract_info_agent(info_docs)
    print(info)
    context = "" 
    if info is not None:
        if info["url"] is not None:
            #now we want to scrape the website
            website_content = scrape_website(info["url"])
            if website_content is not False:
                #now we want to generate questions based on the website content
                context += website_content
        if info["semantical_description"] is not None:
            context += info["semantical_description"]

    print("HERE IS THE CONTEXT: ", context)
    questions = question_agent(target_goal, software)
    print("HERE ARE THE QUESTIONS: ", questions)
    useful_questions = discriminate_questions(questions, software, target_goal)
    print("HERE ARE THE USEFUL QUESTIONS: ", useful_questions)
    useful_info = {
        "initial context": context,
        "question_answer_pairs" : []
    }
    if questions is not None:
        for question in tqdm(questions[:1]):
            answer_from_context = answer_agent_from_context(question, software, target_goal, context)
            answer_from_internet = answer_agent_from_internet(question, software, target_goal)
            useful_info["question_answer_pairs"].append({
                "question": question,
                "answer_from_context": answer_from_context,
                "answer_from_internet": answer_from_internet
            })

    useful_info = str(useful_info)
    prompt = f"The user has the following goal: <user_goal>{target_goal}</user_goal>. The user has the following software: <user_software>{software}</user_software>. Here is some useful context about the software: <user_context>{context}</user_context>. Here is some information about the software: {useful_info}. Please write an extensive summary on how to use the software to achieve the user goal. Consider writing out python code and examples and plans to use the software and the tools necessary to use it and how one would use those tools. Please be thorough."
    system = "You are an assistant that is legally authorized to write code and scrape websites."


    return sonnet.prompt(prompt,max_tokens=4000,system=system).text()
    


