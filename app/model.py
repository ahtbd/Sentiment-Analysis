from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        local_files_only=True
    )

    model.eval()
    return tokenizer, model
