# Reasoning engine 
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

load_dotenv()
@tool
def triple (num:float)-> float:
    """param num : a number to triple
    return : a triple of a param number
    """
    return float(num)*3
tools=[TavilySearch(max_results=1), triple]
llm=ChatGroq(   model="openai/gpt-oss-20b",temperature=0).bind_tools(tools=tools)