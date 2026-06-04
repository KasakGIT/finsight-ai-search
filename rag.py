import requests
import os
import io
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# embedding model - same locally and on deployment
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
CACHE_DIR = "rag_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_annual_report_url(symbol: str) -> str:
    """Fetch latest annual report PDF URL from NSE"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
        'Referer': 'https://www.nseindia.com'
    }
    session = requests.Session()
    session.get('https://www.nseindia.com', headers=headers)
    r = session.get(
        f'https://www.nseindia.com/api/annual-reports?index=equities&symbol={symbol}',
        headers=headers
    )
    data = r.json()
    return data['data'][0]['fileName']

def load_and_chunk_pdf(pdf_url: str) -> list:
    """Download PDF and split into chunks"""
    headers = {'User-Agent': 'Mozilla/5.0'}
    r = requests.get(pdf_url, headers=headers, timeout=30)
    
    # extract text from first 30 pages only (annual reports are huge)
    reader = PdfReader(io.BytesIO(r.content))
    text = ""
    for page in reader.pages[:30]:
        text += page.extract_text() or ""
    
    # split into chunks with overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(text)
    
    # wrap in Document objects
    docs = [Document(page_content=chunk) for chunk in chunks]
    return docs

def get_vectorstore(symbol: str) -> FAISS:
    """Load from cache or build fresh"""
    cache_path = f"{CACHE_DIR}/{symbol}"
    
    if os.path.exists(cache_path):
        print(f"Loading {symbol} from cache...")
        return FAISS.load_local(cache_path, embeddings, allow_dangerous_deserialization=True)
    
    print(f"Fetching annual report for {symbol}...")
    url = get_annual_report_url(symbol)
    print(f"PDF URL: {url}")
    docs = load_and_chunk_pdf(url)
    print(f"Created {len(docs)} chunks")
    
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(cache_path)
    return vectorstore

def search_rag(query: str, symbol: str, k: int = 3) -> str:
    """Search annual report for relevant chunks"""
    try:
        vectorstore = get_vectorstore(symbol)
        results = vectorstore.similarity_search(query, k=k)
        return "\n\n".join([r.page_content for r in results])
    except Exception as e:
        return f"RAG search failed: {str(e)}"

# test
if __name__ == "__main__":
    result = search_rag("What is the total revenue and profit?", "RELIANCE")
    print(result)