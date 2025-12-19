from fastapi import FastAPI
from pydantic import BaseModel
from app.predict import predict_sentiment

app = FastAPI(title="Sentiment Analysis API")

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(req: TextRequest):
    return predict_sentiment(req.text)
