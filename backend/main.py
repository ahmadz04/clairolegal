from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
import openai
import os
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv
load_dotenv()
from openai import RateLimitError
import cohere

app = FastAPI()

# Allow CORS for local frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def analyze_text_with_cohere(text):
    prompt = (
        "Summarize the main focus of this homework/assignment document in less than 20 words. "
        "Respond with only the summary, no extra formatting or explanation.\n\n"
        "Example:\n"
        "Document:\n"
        "Homework 3: Algorithm Analysis. Solve problems 1-4 on dynamic programming and graph algorithms. Due Friday.\n"
        "Summary: Algorithm analysis homework with dynamic programming and graph problems.\n\n"
        "Document:\n"
        f"{text}\n"
        "Summary:"
    )
    co = cohere.Client(os.getenv("COHERE_API_KEY"))
    response = co.generate(
        model="command",
        prompt=prompt,
        max_tokens=30,
        temperature=0.2,
    )
    summary = response.generations[0].text.strip()
    return {"summary": summary}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        text = extract_text_from_pdf(tmp_path)
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text found in PDF.")
        try:
            analysis = analyze_text_with_cohere(text)
        except RateLimitError:
            raise HTTPException(status_code=429, detail="OpenAI API quota exceeded. Please check your OpenAI account.")
        return analysis
    finally:
        os.remove(tmp_path)