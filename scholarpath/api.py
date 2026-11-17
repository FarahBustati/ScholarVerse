"""Thin HTTP API. Run with: uvicorn scholarpath.api:app --reload"""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from scholarpath.graph import build_graph, run_pipeline
from scholarpath.schemas import Opportunity, StudentProfile

app = FastAPI(title="ScholarPath", version="0.1.0")


class RecommendRequest(BaseModel):
    text: str = Field(min_length=20, description="Free-form student background, goals and constraints")


class RecommendResponse(BaseModel):
    profile: StudentProfile
    opportunities: list[Opportunity]
    report_markdown: str
    report_path: str
    errors: list[str]


@lru_cache(maxsize=1)
def get_graph() -> CompiledStateGraph:
    return build_graph()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest) -> RecommendResponse:
    try:
        state = run_pipeline(request.text, graph=get_graph(), interactive=False)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return RecommendResponse(
        profile=state["profile"],
        opportunities=state.get("opportunities", []),
        report_markdown=state["report"],
        report_path=state["report_path"],
        errors=state.get("errors", []),
    )
