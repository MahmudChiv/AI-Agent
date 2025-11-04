from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from agent import run_agent  
import uvicorn


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


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
