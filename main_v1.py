from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from typing import Literal, Annotated
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
import os
from typing import TypedDict
import operator

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
    task="conversational",
    max_new_tokens=1024
)
model=ChatHuggingFace(llm=llm)

search_tool=TavilySearch(max_results=3, tavily_api_key=os.getenv("TAVILY_API_KEY"))

class FinSightState(TypedDict) :
    messages : Annotated[list[AnyMessage], operator.add]
    company : str
    news : list[str]
    financial_data : str
    report : str

tool = [search_tool]
model_with_tools=model.bind_tools(tool)

def news_agent(state: FinSightState):
    company = state["company"]
    response = search_tool.invoke(f"latest news about {company} 2026")
    news_content = [r["content"] for r in response["results"]]
    return {"news" : news_content}


def report_writer(state: FinSightState):
    company = state["company"]
    news = state["news"]
    prompt = f"given the company name {company} and news about the company {news}, generate a structured report for the user"
    response = model.invoke([SystemMessage(content="you are a professional financial analyst report generator"), HumanMessage(content=prompt)])
    return {"report" : response.content}

graph = StateGraph(FinSightState)
graph.add_node("news_agent", news_agent)
graph.add_node("report_writer", report_writer) 
graph.add_edge(START, "news_agent")
graph.add_edge("news_agent", "report_writer")
graph.add_edge("report_writer", END)

agent = graph.compile()
result = agent.invoke({
    "company" : "TCS", 
    "messages" : [],
    "news" : [],
    "financial_data" : "",
    "report" : ""
})
print(result["report"])

# # test it
# result = news_agent({"company": "Reliance Industries", "messages": [], "news": [], "financial_data": "", "report": ""})
# result = report_writer({"company": "Reliance Industries", "messages": [], "news": [], "financial_data": "", "report": ""})
