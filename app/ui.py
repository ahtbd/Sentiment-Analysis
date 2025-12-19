import os
import requests
import gradio as gr
from app.vector_store import SentimentVectorStore

# ----------------------------
# Environment cleanup (proxy-safe)
# ----------------------------
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("ALL_PROXY", None)
os.environ.pop("all_proxy", None)

# Offline mode (safety)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ----------------------------
# Configuration
# ----------------------------
API_URL = "http://sentiment-api:7860/predict"

vector_db = SentimentVectorStore()
history = []  # store last 5 inputs

# ----------------------------
# UI → API call
# ----------------------------
def analyze_sentiment_ui(text):
    if not text or text.strip() == "":
        return "", 0, "", "", ""

    try:
        response = requests.post(
            API_URL,
            json={"text": text},
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
    except Exception as e:
        return (
            "<h3 style='color:red'>API Error</h3>",
            0,
            "",
            "",
            str(e)
        )

    sentiment = result["sentiment"]
    confidence = result["confidence"]

    # Uncertain handling
    display_sentiment = sentiment
    if confidence < 0.6:
        display_sentiment = "UNCERTAIN"

    color = "green" if sentiment == "POSITIVE" else "red"

    sentiment_html = f"""
    <h2 style="text-align:center; color:{color};">
        {display_sentiment}
    </h2>
    """

    # Update history
    history.append(f"{text[:40]}... → {sentiment} ({confidence})")
    last_history = "\n".join(history[-5:])

    # Vector DB similarity search
    similar = vector_db.similarity_search(text, top_k=3)
    similar_texts = ""
    for doc, meta in zip(similar["documents"][0], similar["metadatas"][0]):
        similar_texts += f"- {doc[:80]}... | {meta['sentiment']}\n"

    return sentiment_html, confidence, last_history, similar_texts, ""

# ----------------------------
# Clear UI
# ----------------------------
def clear_all():
    return "", 0, "", "", ""

# ----------------------------
# Semantic Search UI
# ----------------------------
def semantic_search_ui(query):
    if not query or query.strip() == "":
        return ""

    results = vector_db.similarity_search(query, top_k=5)
    output = ""
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        output += f"- {doc[:80]}... | Sentiment: {meta['sentiment']}\n"

    return output

# ----------------------------
# Example handler
# ----------------------------
def example_click(text):
    return analyze_sentiment_ui(text)

# ----------------------------
# UI Layout
# ----------------------------
with gr.Blocks(title="Sentiment & Semantic Search System") as demo:

    gr.Markdown(
        """
        # Sentiment Analysis & Semantic Search System  
        **Transformer-based NLP system with REST API + Vector Database**
        
        - UI → calls backend API  
        - API → runs Transformer model  
        - Vector DB → ChromaDB  
        """
    )

    # ------------------------
    # Sentiment Section
    # ------------------------
    gr.Markdown("## Sentiment Analysis")

    input_text = gr.Textbox(
        label="Input Text",
        placeholder="Type text or click an example below...",
        lines=4
    )

    with gr.Row():
        analyze_btn = gr.Button("Analyze Sentiment", variant="primary")
        clear_btn = gr.Button("Clear")

    sentiment_output = gr.HTML(label="Predicted Sentiment")

    confidence_bar = gr.Slider(
        minimum=0,
        maximum=1,
        step=0.01,
        label="Confidence Score"
    )

    with gr.Row():
        history_box = gr.Textbox(
            label="History (Last Inputs)",
            lines=5
        )
        similar_box = gr.Textbox(
            label="Similar Past Texts (Vector DB)",
            lines=5
        )

    gr.Markdown("### Example Texts (Click to Analyze)")
    with gr.Row():
        ex1 = gr.Button("I love this product")
        ex2 = gr.Button("This is the worst service")
        ex3 = gr.Button("The movie was okay")

    analyze_btn.click(
        fn=analyze_sentiment_ui,
        inputs=input_text,
        outputs=[
            sentiment_output,
            confidence_bar,
            history_box,
            similar_box,
            input_text
        ]
    )

    clear_btn.click(
        fn=clear_all,
        outputs=[
            sentiment_output,
            confidence_bar,
            history_box,
            similar_box,
            input_text
        ]
    )

    ex1.click(
        fn=lambda: example_click("I love this product"),
        outputs=[sentiment_output, confidence_bar, history_box, similar_box, input_text]
    )

    ex2.click(
        fn=lambda: example_click("This is the worst service"),
        outputs=[sentiment_output, confidence_bar, history_box, similar_box, input_text]
    )

    ex3.click(
        fn=lambda: example_click("The movie was okay"),
        outputs=[sentiment_output, confidence_bar, history_box, similar_box, input_text]
    )

    # ------------------------
    # Semantic Search Section
    # ------------------------
    gr.Markdown("## Semantic Search (Vector Database)")

    search_input = gr.Textbox(
        label="Search Similar Stored Texts",
        placeholder="Example: bad customer service"
    )

    search_btn = gr.Button("Search")

    search_output = gr.Textbox(
        label="Semantic Search Results",
        lines=6
    )

    search_btn.click(
        fn=semantic_search_ui,
        inputs=search_input,
        outputs=search_output
    )

    gr.Markdown(
        """
        **Notes:**  
        - UI communicates with backend via REST API  
        - Sentiment inference runs only in API container  
        - Vector search uses ChromaDB  
        """
    )

# ----------------------------
# Launch UI
# ----------------------------
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
