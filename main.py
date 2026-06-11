from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from typing import Literal, Annotated
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
import os
import yfinance as yf
from typing import TypedDict
import operator
from rag import search_rag
import time
#from langchain_ollama import ChatOllama
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
    task="conversational",
    max_new_tokens=1024,
    provider="novita" 
)
model = ChatHuggingFace(llm=llm)


#model = ChatOllama(model="llama3.2")

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

def call_model_with_retry(model, messages, retries=3):
    for attempt in range(retries):
        try:
            return model.invoke(messages)
        except Exception as e:
            if attempt < retries - 1:
                print(f"Attempt {attempt+1} failed, retrying in 3s...")
                time.sleep(3)
            else:
                raise e

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

    intent = "explain"
    company = "None"
    
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
    return "explain_node"
    pass

def explain_node(state: FinSightState):
    query = state["query"]
    response = model.invoke([
        SystemMessage(content="You are a helpful financial assistant. Always answer directly. Never refuse."),
        *state["messages"],        # conversation history
        HumanMessage(content=query)
    ])
    return {"report": response.content}

def retrieve_node(state: FinSightState):
    company = state["company"]
    
    try:
        search_results = yf.Search(company, news_count=0).quotes
        print("COMPANY:", company)
        print("SEARCH RESULTS:", search_results)
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
        
    except Exception as e:
        print("SEARCH ERROR:", str(e))
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
        prompt = f"""Company: {company}
                Stock Data: {stock_data}
                News: {news}
                Analyst Opinions: {analyst}
                Annual Report Excerpts: {rag_data}

                Write a concise investment report using EXACTLY this format with proper newlines:

                ## Financial Summary
                - point 1
                - point 2

                ## Recent Developments
                - point 1
                - point 2

               ## Analyst Consensus
               - point 1

               ## Key Risks
               - point 1
               - point 2

               ## Recommendation
               - Buy/Sell/Hold with target price

                IMPORTANT: Each section MUST be on its own line. Keep under 300 words."""

        response = model.invoke([
            SystemMessage(content="""
                        You are a professional financial analyst writing research reports. 
                        You ALWAYS provide analysis. Never refuse or say you can't provide advice.
                        Write objective research reports with data."""),
            HumanMessage(content=prompt)
        ])
        return {"report": response.content}
    
def evaluator(state: FinSightState):
    report = state["report"]
    intent = state["intent"]
    
    # explain intent - just check it's not empty
    if intent == "explain":
        is_good = len(report) > 10
        print(f"Evaluation: {'PASSED' if is_good else 'FAILED'}")
        return {"is_good_enough": is_good}
    
    # retrieve intent - just check it has data
    if intent == "retrieve":
        is_good = len(report) > 10
        print(f"Evaluation: {'PASSED' if is_good else 'FAILED'}")
        return {"is_good_enough": is_good}
    
    # analyze intent - full checks
    issues = []
    if len(report) < 300:
        issues.append("too short")
    
    required_sections = ["financial", "risk", "outlook", "recommendation"]
    missing = [s for s in required_sections if s.lower() not in report.lower()]
    if len(missing) > 2:
        issues.append(f"missing sections: {missing}")
    
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