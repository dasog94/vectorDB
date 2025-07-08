from fastapi import APIRouter, Query
from app.function.adapter.output.repository import FunctionRepository
import pandas as pd

function_router = APIRouter()
repository = FunctionRepository()


@function_router.get("/search/hybrid")
async def hybrid_search(
    keyword: str = Query("", description="Keyword to search for"),
    k: int = Query(5, description="Number of results to return"),
    alpha: float = Query(0.7, description="Alpha value for hybrid search")
):
    result = repository.search_hybrid(keyword, k, alpha)

    return {"result": result}

@function_router.post("/load-data")
async def load_data():
    repository.load_data()

    return {"result": "ok"}