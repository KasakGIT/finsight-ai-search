from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from typing import Literal, Annotated
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import yfinance as yf
from typing import TypedDict
import operator
from rag import search_rag
from langchain_ollama import ChatOllama

load_dotenv()

model = ChatOllama(model="llama3.2")

search_tool=TavilySearch(max_results=3, tavily_api_key=os.getenv("TAVILY_API_KEY"))

class FinSightState(TypedDict) :
    messages : Annotated[list[AnyMessage], operator.add]
    query : str
    intent : str
    stock_data : str
    analyst_opinion : list[str]
    company : str
    news : list[str]
    report : str
    is_good_enough : bool
    rag_data: str

def intent_classifier(state: FinSightState):
    query = state["query"]
    response = model.invoke([
        SystemMessage(content="""You are a financial query classifier.
Classify the query as ONE of:
- explain: conceptual question, no tools needed
- retrieve: fetch specific data point
- analyze: multi-step research, comparison, investment advice

Also extract the full official company name as listed on stock excahnges 
Examples: 
- "Reliance" → "Reliance Industries Limited"
- "TCS" → "Tata Consultancy Services Limited"
- "Infosys" → "Infosys Limited"(write None if explain).
Reply EXACTLY in this format with nothing else:
intent: analyze
company: Reliance Industries Limited"""),
        HumanMessage(content=query)
    ])
    
    
    response_text = response.content.strip()
    
    for line in response_text.split("\n"):
        if line.startswith("intent:"):
            intent = line.split(": ")[1].strip()
        elif line.startswith("company:"):
            company = line.split(": ")[1].strip()
    
    return {"intent": intent, "company": company}


def route_query(state: FinSightState) -> Literal["explain_node", "retrieve_node", "analyze_node"]:
    intent = state["intent"]
    if intent=="explain": return "explain_node"
    if intent=="retrieve": return "retrieve_node"
    if intent=="analyze": return "analyze_node" 
    pass

def explain_node(state: FinSightState) :
    query = state["query"]
    response = model.invoke([HumanMessage(content=query)])
    return { "report" : response.content}

def retrieve_node(state: FinSightState):
    company = state["company"]
    
    try:
        search_results = yf.Search(company, news_count=0).quotes
        if not search_results:
            # try with just first word
            short_name = company.split()[0]
            search_results = yf.Search(short_name, news_count=0).quotes
    # prefer RELIANCE.NS over RELINFRA.NS
        indian_stocks = [q for q in search_results 
                     if q.get('symbol', '').endswith('.NS') 
                     and q.get('quoteType') == 'EQUITY']
    
        if indian_stocks:
            ticker = indian_stocks[0]['symbol']
        else:
            ticker = search_results[0]['symbol']
        
    except:
        ticker = None

    print("TICKER FOUND:", ticker)

    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        data = f"Price: {info.get('currentPrice', 'N/A')}, MarketCap: {info.get('marketCap', 'N/A')}, PE: {info.get('trailingPE', 'N/A')}"
    except:
        data = f"Could not fetch data for {company}"
    
    return {"stock_data": data}

def analyze_node(state: FinSightState):
    company = state["company"]

    try:
        search_results = yf.Search(company, news_count=0).quotes
        if not search_results:
            # try with just first word
            short_name = company.split()[0]
            search_results = yf.Search(short_name, news_count=0).quotes
    # prefer RELIANCE.NS over RELINFRA.NS
        indian_stocks = [q for q in search_results 
                     if q.get('symbol', '').endswith('.NS') 
                     and q.get('quoteType') == 'EQUITY']
    
        if indian_stocks:
            ticker = indian_stocks[0]['symbol']
        else:
            ticker = search_results[0]['symbol']
        
    except:
        ticker = None

    print("TICKER FOUND:", ticker)

    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        data = f"Price: {info.get('currentPrice', 'N/A')}, MarketCap: {info.get('marketCap', 'N/A')}, PE: {info.get('trailingPE', 'N/A')}"
    except:
        data = f"Could not fetch data for {company}"
    
    # search 1 - news
    news_response = search_tool.invoke(f"latest news about {company} 2026")
    news = [r["content"] for r in news_response["results"]]
    # add temporarily to analyze_node
    
    # search 2 - analyst opinions
    analyst_response = search_tool.invoke(f"give analyst opinions about {company} 2026")
    analyst_opinion = [r["content"] for r in analyst_response["results"]]
    
    rag_data = search_rag(
        query=state["query"], 
        symbol=ticker.replace(".NS", "") if ticker else company.upper()
    )

    return {"news": news, "analyst_opinion": analyst_opinion, "stock_data": data, "rag_data": rag_data}

def report_writer(state: FinSightState):
    intent = state["intent"]
    
    if intent == "explain":
        return {}  # already in state["report"] from explain_node
    
    elif intent == "retrieve":
        # just format stock_data, no LLM needed
        return {"report": state["stock_data"]}
    
    elif intent == "analyze":
        company = state["company"]
        news = state["news"]
        analyst = state["analyst_opinion"]
        stock_data = state["stock_data"]
        rag_data = state["rag_data"]
        prompt = f"""You are a financial analyst writing a research report (NOT personal advice).
                Company: {company}
                Stock Data: {stock_data}
                News: {news}
                Analyst Opinions: {analyst}
                Annual Report Excerpts: {rag_data}

                Write a structured research report with sections:
                1. Financial Summary
                2. Recent Developments  
                3. Analyst Consensus
                4. Key Risks
                5. Outlook
                Write a structured investment report using ALL the above data sources.
                Prioritize exact figures from the annual report excerpts."""
        response = model.invoke([
            SystemMessage(content="You are a professional financial analyst."),
            HumanMessage(content=prompt)
        ])
        return {"report": response.content}
    
def evaluator(state: FinSightState):
    report = state["report"]
    intent = state["intent"]
    issues = []
    
    # check 1 - minimum length
    if len(report) < 300:
        issues.append("too short")
    
    # check 2 - has required sections
    required_sections = ["financial", "risk", "outlook", "recommendation"]
    missing = [s for s in required_sections if s.lower() not in report.lower()]
    if len(missing) > 2:
        issues.append(f"missing sections: {missing}")
    
    # check 3 - has actual numbers (good report has data)
    import re
    numbers = re.findall(r'₹[\d,]+|[\d]+\.[\d]+%|\$[\d,]+', report)
    if intent == "analyze" and len(numbers) < 2:
        issues.append("insufficient data points")
    
    # check 4 - not an error/refusal
    refusal_phrases = ["i cannot", "i'm unable", "i don't have"]
    if any(p in report.lower() for p in refusal_phrases):
        issues.append("llm refused to answer")
    
    is_good = len(issues) == 0
    print(f"Evaluation: {'PASSED' if is_good else 'FAILED - ' + str(issues)}")
    
    return {"is_good_enough": is_good}

def should_continue(state: FinSightState) -> Literal["END", "report_writer"] :
    good_enough = state["is_good_enough"]
    if(good_enough): return END
    return "report_writer"


graph = StateGraph(FinSightState)

graph.add_node("explain_node", explain_node)
graph.add_node("retrieve_node", retrieve_node)
graph.add_node("analyze_node", analyze_node)
graph.add_node("evaluator", evaluator)
graph.add_node("report_writer", report_writer)
graph.add_node("intent_classifier", intent_classifier)

graph.add_edge(START, "intent_classifier")
graph.add_conditional_edges("intent_classifier", route_query, ["explain_node", "retrieve_node", "analyze_node"])
graph.add_edge("retrieve_node", "report_writer")
graph.add_edge("analyze_node", "report_writer")
graph.add_edge("explain_node", "report_writer")
graph.add_edge("report_writer", "evaluator")
graph.add_conditional_edges("evaluator", should_continue, [END, "report_writer"])


agent = graph.compile()

result = agent.invoke({
    "query": "Should I invest in Dixon Technologies?",
    "messages": [],
    "intent": "",
    "company": "",
    "stock_data": "",
    "news": [],
    "analyst_opinion": "",
    "report": "",
    "is_good_enough": False
} , config={"recursion_limit": 5})

print(result["report"])

# print("INTENT:", result["intent"])
# print("COMPANY:", result["company"])
# print("STOCK DATA:", result["stock_data"])
# print("REPORT:", result["report"])