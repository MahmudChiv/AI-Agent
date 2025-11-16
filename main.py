from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from dotenv import load_dotenv

import os

from models.a2a import JSONRPCRequest, JSONRPCResponse, TaskResult, TaskStatus, Artifact, MessagePart, A2AMessage

from agents.agent import run_agent  
load_dotenv()
from pydantic import BaseModel
from typing import List, Optional
import uvicorn


app = FastAPI()

class Message(BaseModel):
    role: str
    content: str

class Query(BaseModel):
    question: str
    history: Optional[List[Message]] = []

@app.post("/a2a/dataGen")
async def a2a_endpoint(request: Request):
    """Main A2A endpoint for Datagen"""
    try:
        body = await request.json()
        if body.get("jsonrpc") != "2.0" or "id" not in body:
            return JSONResponse(
                status_code=400,
                content={
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "error": {
                        "code": -32600,
                        "message": "Invalid Request: jsonrpc must be '2.0' and id is required"
                    }
                }
            )
        
        rpc_request = JSONRPCRequest(**body)

        messages = []
        context_id = None
        task_id = None
        config = None

        if rpc_request.method == "message/send":
            messages = [rpc_request.params.message]
            config = rpc_request.params.configuration
        elif rpc_request.method == "execute":
            messages = rpc_request.params.messages
            context_id = rpc_request.params.contextId
            task_id = rpc_request.params.taskId

        result = await run_agent.process_messages(
            messages=messages,
            context_id=context_id,
            task_id=task_id,
            config=config
        )

        response = JSONRPCResponse(
            id=rpc_request.id,
            result=result
        )

        return response.model_dump()
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "jsonrpc": "2.0",
                "id": body.get("id") if "body" in locals() else None,
                "error": {
                    "code": -32603,
                    "message": "Internal error",
                    "data": {"details": str(e)}
                }
            }
        )
async def ask_agent(query: Query):
    result = await run_agent(query.question, query.history)
    return {"response": result}

@app.get("/")
def root():
    return {"message": "AI Agent API is running 🚀"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    uvicorn.run(app, host="0.0.0.0", port=port)