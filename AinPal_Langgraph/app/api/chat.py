from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.core.pipeline import process_chat

router = APIRouter()

class ChatRequest(BaseModel):
    user_message: str

class ChatResponse(BaseModel):
    bot_response: str

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        response = process_chat(request.user_message)
        return ChatResponse(bot_response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))