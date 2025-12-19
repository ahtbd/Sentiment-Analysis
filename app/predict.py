import torch
import torch.nn.functional as F
from app.model import load_model
from app.preprocess import clean_text
from app.vector_store import SentimentVectorStore

tokenizer, model = load_model()
vector_db = SentimentVectorStore()

LABELS = {0: "NEGATIVE", 1: "POSITIVE"}

def predict_sentiment(text: str):
    text = clean_text(text)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)

    confidence, predicted_class = torch.max(probs, dim=1)
    sentiment = LABELS[predicted_class.item()]
    conf_score = round(confidence.item(), 4)

    # Store in vector database
    vector_db.add_record(
        text=text,
        sentiment=sentiment,
        confidence=conf_score
    )

    return {
        "sentiment": sentiment,
        "confidence": conf_score
    }
