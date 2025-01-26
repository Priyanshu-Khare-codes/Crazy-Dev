from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import google.generativeai as genai
from typing_extensions import TypedDict
import enum

# Load environment variables
load_dotenv()

# Configure Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, or specify a list of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Enums
class Tag(enum.Enum):
    Positive = "Positive"
    Negative = "Negative"
    Neutral = "Neutral"

class Category(enum.Enum):
    Hate_Speech = "Hate Speech"
    Violence = "Violence"
    Bullying = "Bullying"
    Harassment = "Harassment"
    Misinformation = "Misinformation"
    Spam = "Spam"
    Fake_News = "Fake News"
    Nota = "Nota"

# Pydantic Model for Request and Response Validation
class SentimentAnalysis(BaseModel):
    tag: Tag
    category: Category
    description: str

class AnalyzeRequest(BaseModel):
    text: str

# API Endpoint
@app.post("/analyze", response_model=SentimentAnalysis)
async def analyze_text(request: AnalyzeRequest):
    try:
        # Call the Generative AI model
        result = model.generate_content(
            f"Analyse the text and flag thembased on tag, category & description of it : {request.text}",
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=SentimentAnalysis
            )
        )
        
        # Parse the response
        response_data = result.text
        return SentimentAnalysis.parse_raw(response_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing text: {str(e)}")

# Health Check Endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}
