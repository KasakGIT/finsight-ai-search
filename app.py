import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage
from main import agent
import time
import markdown as md

custom_css = """
* { font-family: -apple-system, BlinkMacSystemFont, 'Inter', sans-serif !important; }
body, .gradio-container { background: #f8f9fb !important; }
.gradio-container { max-width: 900px !important; margin: 0 auto !important; padding: 0 !important; }
#msg-input textarea { background: #f8fafc !important; border: 0.5px solid #e2e8f0 !important; border-radius: 10px !important; font-size: 14px !important; color: #0f172a !important; padding: 10px 14px !important; resize: none !important; }
#send-btn { background: #0f172a !important; color: #ffffff !important; border: none !important; border-radius: 8px !important; font-size: 14px !important; font-weight: 500 !important; padding: 10px 20px !important; }
#chatbot p { font-size: 13px !important; }
#chatbot h2 { font-size: 15px !important; font-weight: 600 !important; margin: 8px 0 4px 0 !important; border-bottom: 1px solid #e2e8f0 !important; padding-bottom: 4px !important; }
#chatbot h3 { font-size: 14px !important; font-weight: 600 !important; }
#chatbot ul { font-size: 13px !important; padding-left: 16px !important; }
#chatbot li { margin: 2px 0 !important; }
"""

def chat(message, history):
    # show user message + thinking indicator
    history = history + [{"role": "user", "content": message}]
    history = history + [{"role": "assistant", "content": "..."}]
    yield history, ""

    # build message history for LLM context
    messages = []
    for item in history[:-2]:
        if item["role"] == "user":
            messages.append(HumanMessage(content=item["content"]))
        elif item["role"] == "assistant":
            messages.append(AIMessage(content=item["content"]))

    result = agent.invoke({
        "query": message,
        "messages": messages,
        "intent": "",
        "company": "",
        "stock_data": "",
        "news": [],
        "analyst_opinion": [],
        "report": "",
        "rag_data": "",
        "is_good_enough": False
    }, config={"recursion_limit": 25})

    # convert markdown to HTML
    report_html = md.markdown(result["report"])

    # stream line by line (not word by word - HTML tags break otherwise)
    output = ""
    history[-1]["content"] = ""
    for line in report_html.split("\n"):
        output += line + "\n"
        history[-1]["content"] = output
        yield history, ""
        time.sleep(0.05)

with gr.Blocks(title="FinSight", css=custom_css) as demo:
    gr.HTML("""
    <div style="background:#fff; border-bottom:0.5px solid #e5e7eb; padding:16px 28px; display:flex; align-items:center; justify-content:space-between;">
        <div>
            <span style="font-size:20px; font-weight:600; color:#0f172a; letter-spacing:-0.3px;">Fin<span style="color:#ea580c;">Sight</span></span>
            <span style="font-size:12px; color:#94a3b8; margin-left:10px;">AI Financial Research Assistant</span>
        </div>
        <div style="font-size:12px; color:#64748b; background:#f8fafc; border:0.5px solid #e2e8f0; border-radius:6px; padding:5px 10px;">
            🟢 Qwen 72B · HuggingFace
        </div>
    </div>
    """)

    chatbot = gr.Chatbot(
        elem_id="chatbot",
        height=500,
        show_label=False,
        sanitize_html=False,
        type='messages'
    )

    with gr.Row():
        msg = gr.Textbox(
            elem_id="msg-input",
            placeholder="Ask about any Indian stock…",
            show_label=False,
            scale=5,
            container=True
        )
        send = gr.Button("Send →", elem_id="send-btn", scale=1)

    gr.Examples(
        examples=[
            "Should I invest in Reliance Industries?",
            "What is TCS current stock price?",
            "What is a P/E ratio?",
            "HDFC Bank vs ICICI Bank",
        ],
        inputs=msg,
        label="",
    )

    msg.submit(chat, [msg, chatbot], [chatbot, msg])
    send.click(chat, [msg, chatbot], [chatbot, msg])

demo.launch()