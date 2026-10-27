"""
main.py - FastAPI application for RESEARCHLY.

Exposes:
- POST /api/research - Runs evidence-backed research agent
- GET  /api/health   - Service health and API configuration status
"""

import os
import sys
import time
import logging
from typing import Optional, List
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# Load environment variables from .env file
load_dotenv()

from search_tool import TavilySearchTool
from research_agent import ResearchAgent, ResearchReport, AgentError

# Configure structured logging (ensure secrets are never logged)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("researchly.api")


# --- Lifespan & App Setup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup validation
    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    logger.info("Starting RESEARCHLY Backend API...")
    logger.info(f"Tavily Search Configured: {'YES' if tavily_key else 'NO'}")
    logger.info(f"Gemini LLM Configured:    {'YES' if gemini_key else 'NO'}")
    yield
    logger.info("Shutting down RESEARCHLY Backend API...")


app = FastAPI(
    title="RESEARCHLY API",
    description="Tool-Using AI Research Agent Backend powered by Tavily Search and Google Gemini",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration
# Allow frontend origins for local development and configurable origins
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
if allowed_origins_env:
    allowed_origins = [orig.strip() for orig in allowed_origins_env.split(",") if orig.strip()]
else:
    allowed_origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080"
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request and Response Models ---
class ResearchRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=600,
        description="The research topic or question to investigate.",
        json_schema_extra={"example": "How is AI transforming healthcare?"}
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Research question cannot be empty or whitespace only.")
        return stripped


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    search_configured: bool
    llm_configured: bool


# --- Exception Handlers ---
@app.exception_handler(AgentError)
async def agent_exception_handler(request: Request, exc: AgentError):
    # Sanitize message to never leak raw keys if somehow present
    safe_msg = exc.message.replace(os.getenv("TAVILY_API_KEY", "___"), "[REDACTED]")
    safe_msg = safe_msg.replace(os.getenv("GEMINI_API_KEY", "___"), "[REDACTED]")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": True, "detail": safe_msg}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled server exception:")
    return JSONResponse(
        status_code=500,
        content={"error": True, "detail": "An unexpected internal server error occurred."}
    )


# --- API Routes ---
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint reporting API status and tool availability."""
    tavily_key = bool(os.getenv("TAVILY_API_KEY", "").strip())
    gemini_key = bool(os.getenv("GEMINI_API_KEY", "").strip())
    return HealthResponse(
        status="healthy",
        service="RESEARCHLY AI Agent Backend",
        version="1.0.0",
        search_configured=tavily_key,
        llm_configured=gemini_key
    )


@app.post("/api/research", response_model=ResearchReport)
async def conduct_research(request: ResearchRequest):
    """
    Conducts live tool-based research:
    1. Searches the internet via Tavily API.
    2. Retrieves up to 5 real sources.
    3. Synthesizes evidence with Google Gemini.
    4. Enforces numbered citations matching sources.
    5. Returns validated research report.
    """
    req_start = time.perf_counter()
    # STAGE 1 DIAGNOSTIC LOG
    logger.info(
        f"[STAGE 1: RESEARCH REQUEST RECEIVED] Question: '{request.question[:60]}' "
        f"(length: {len(request.question)} chars)"
    )

    search_tool = TavilySearchTool()
    agent = ResearchAgent(search_tool=search_tool)

    report = await agent.conduct_research(request.question)
    total_elapsed = time.perf_counter() - req_start

    # STAGE 6 DIAGNOSTIC LOG
    logger.info(
        f"[STAGE 6: RESEARCH RESPONSE RETURNED] Question: '{request.question[:60]}', "
        f"Total duration: {total_elapsed:.2f}s, Sources: {len(report.sources)}, "
        f"Model: {report.model_used or 'N/A'}"
    )
    return report


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("main:app", host=host, port=port, reload=True)
