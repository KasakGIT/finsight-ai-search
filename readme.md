# 📈 FinSight – AI Financial Research Assistant

FinSight is an AI-powered financial research assistant that helps users analyze companies, fetch real-time stock data, and generate investment insights using a multi-agent workflow. It combines LLMs, retrieval systems, and financial APIs into a single intelligent pipeline.

---

## 🚀 Features

- 🔍 **Company Intelligence Agent**
  - Understands user queries about stocks and companies
  - Extracts intent (stock data, news, analysis)

- 📊 **Real-Time Stock Data**
  - Fetches live market data using `yfinance`
  - Displays price, market trends, and key metrics

- 🧠 **RAG-Based Financial Analysis**
  - Processes annual reports (PDFs)
  - Uses FAISS vector database for semantic search
  - Retrieves relevant financial insights

- 🌐 **Web Search Integration**
  - Uses Tavily API for latest financial news
  - Enhances answers with real-time context

- 🤖 **Multi-Agent Workflow (LangGraph)**
  - Structured pipeline for:
    - Query understanding
    - Data fetching
    - Analysis generation
    - Final response synthesis

- 🖥️ **Interactive UI**
  - Built with Gradio
  - Chat-based interface for financial queries

---

## 🏗️ System Architecture

User Query  
→ Intent Classification (LLM)  
→ LangGraph Router  
→ Tool Selection:
- Stock Data Agent (`yfinance`)
- News Search Agent (`Tavily`)
- RAG Agent (FAISS + embeddings)

→ Response Generator (LLM)  
→ Final Answer (Gradio UI)

---

## 🛠️ Tech Stack

- Python
- LangChain / LangGraph
- Hugging Face Transformers
- FAISS (Vector Database)
- yfinance
- Tavily Search API
- SentenceTransformers
- Gradio

---

---

## ⚙️ How It Works

1. User enters a financial query (e.g. *“Should I invest in TCS?”*)
2. System detects intent:
   - Stock data → `yfinance`
   - Analysis → RAG pipeline
   - News → Tavily search
3. Relevant data is fetched
4. LLM generates final structured response
5. Output is displayed in Gradio UI

---

## 📊 Example Queries

- What is the current stock price of Reliance?
- Analyze TCS annual performance
- Should I invest in Infosys?
- Latest news about HDFC Bank

---

## ⚠️ Known Limitations

- Stock APIs may face rate limits on cloud deployments
- Initial startup may be slow due to model loading
- External APIs (Tavily/Yahoo Finance) may occasionally fail

---

## 🔮 Future Improvements

- Portfolio tracking dashboard
- Sentiment analysis on financial news
- Multi-stock comparison mode
- Response caching for faster performance
- Mobile-friendly UI

---

## 📌 Author

Built as a full-stack AI finance project integrating:
- LLMs
- RAG pipelines
- Real-time financial APIs
- Multi-agent orchestration

---

## 🏁 Status

✔️ Fully functional  
✔️ Deployed on Hugging Face Spaces  
✔️ End-to-end AI workflow implemented
