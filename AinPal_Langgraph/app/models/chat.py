from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    user_id: str
    message: str
    context: Optional[List[str]] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    timestamp: str