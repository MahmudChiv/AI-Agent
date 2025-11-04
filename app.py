from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from main import run_agent  # your AI logic

app = FastAPI()

class Message(BaseModel):
    role: str
    content: str

class Query(BaseModel):
    question: str
    history: Optional[List[Message]] = []

@app.post("/")
async def ask_agent(query: Query):
    result = await run_agent(query.question, query.history)
    return {"response": result}

@app.get("/")
def root():
    return {"message": "AI Agent API is running 🚀"}
