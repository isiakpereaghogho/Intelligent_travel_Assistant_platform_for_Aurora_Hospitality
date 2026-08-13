import sys
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.agents.aggregator_agent import agentic_rag_answer
from src.database.memory_db import get_memory_db
from src.evaluators.agent_evaluator import RAGEvaluator
from src.chains.agent_chains import get_llm
from src.exception import CustomException
from src.logger import setup_logger


logger = setup_logger()

app = FastAPI(title="Aurora Hotel Chatbot API")

# CORS CONFIGURATION
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501"
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

# PYDANTIC REQUEST / RESPONSE MODELS

class ChatRequest(BaseModel):
    question: str
    guest_type: str
    loyalty: str
    city: str
    session_id: str

class ChatResponse(BaseModel):
    answer: str
    session_id: str
    success: bool
    evaluation: Optional[Dict[str, Any]] = None

class ClearMemoryRequest(BaseModel):
    session_id: str

class EvaluationRequest(BaseModel):
    response: str
    question: str
    context: Optional[str] = "Hotel policy context"

# APPLICATION COMPONENTS

memory_db = get_memory_db()
llm = get_llm()
evaluator = RAGEvaluator(llm=llm)

# ROOT ENDPOINT

@app.get("/")
async def root():
    logger.info("Root endpoint accessed.")
    return {"message": "Welcome to the Aurora Hotel Chatbot API!"}

# CHAT ENDPOINT

@app.post("/chat",response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        logger.info("Received chat request - Session: %s, Question: %s", request.session_id, request.question[:50])

        result = await agentic_rag_answer(
            question=request.question,
            guest_type=request.guest_type,
            loyalty=request.loyalty,
            city=request.city,
            session_id=request.session_id,
            evaluate=True
        )

        session_id = result.get("session_id",request.session_id)

        logger.info("Generated chat response - Session: %s", session_id)

        return ChatResponse(
            answer=result.get("text") or result.get("answer") or result.get("result"),
            session_id=session_id,
            success=True,
            evaluation=result.get("evaluation")
        )

    except Exception as e:
        logger.exception("Custom exception in chat endpoint.")
        raise HTTPException(status_code=400, detail=str(e)) from e

    except Exception as e:
        logger.exception("Unexpected error in chat endpoint.")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.") from e

# CONVERSATION HISTORY ENDPOINT

@app.get("/history/{session_id}")
async def get_history(session_id: str):
    try:
        logger.info("Fetching history for session: %s", session_id)
        history_tuples = (memory_db.get_chat_history_tuples(session_id))
        history_text = (memory_db.get_chat_history_text(session_id))
        return {
            "session_id": session_id,
            "history": history_tuples,
            "history_text": history_text,
            "success": True
        }
    except Exception as e:
        logger.exception("Custom exception in history endpoint.")
        raise HTTPException(status_code=400, detail=str(e)) from e
    
    except Exception as e:
        logger.exception("Unexpected error in history endpoint.")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.") from e

# CLEAR MEMORY ENDPOINT

@app.post("/clear_memory")
async def clear_memory(request: ClearMemoryRequest):
    try:
        logger.info("Clearing memory for session: %s", request.session_id)
        memory_db.clear_memory(request.session_id)

        return {
            "message": "Memory cleared",
            "session_id": request.session_id,
            "success": True
        }

    except Exception as er:
        logger.exception("Custom exception in clear-memory endpoint.")
        raise HTTPException(status_code=400,detail=str(e)) from e

    except Exception as e:
        logger.exception("Unexpected error in clear-memory endpoint.")
        raise HTTPException(status_code=500,detail="An unexpected error occurred.") from e

# EVALUATION ENDPOINT
@app.post("/evaluate",response_model=Dict[str, Any])
async def evaluate(request: EvaluationRequest):
    try:
        logger.info("Evaluating response for question: %s", request.question[:50])
        quality = evaluator.evaluate_response_quality(
            response=request.response,
            question=request.question,
            context=request.context or "",
            verbose=True
        )
        return {
            "faithfulness": quality["faithfulness"],
            "relevance": quality["relevance"],
            "overall": quality["overall"],
            "success": True
            }
    except CustomException as error:
        logger.exception("Custom exception in evaluation endpoint.")
        raise HTTPException(status_code=400, detail=str(error)) from error

    except Exception as e:
        logger.exception("Unexpected error in evaluation endpoint.")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.") from e

# HEALTH CHECK ENDPOINT
@app.get("/health")
async def health_check():
    logger.debug("Health check endpoint accessed.")

    return {"status": "healthy"}

# RUN API DIRECTLY
if __name__ == "__main__":
    logger.info("Starting Aurora Hotel Chatbot API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)