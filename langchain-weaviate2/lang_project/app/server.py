from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware import Middleware

from app.function.adapter.input.function import function_router


def init_routers(app_: FastAPI) -> None:
    app_.include_router(function_router)

def make_middleware() -> list[Middleware]:
    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        ),
    ]
    return middleware

def create_app() -> FastAPI:
    app_ = FastAPI(
        title="Langchain Weaviate API",
        description="Langchain Weaviate API",
        version="1.0.0",
        middleware=make_middleware(),
    )
    init_routers(app_=app_)
    return app_


app = create_app()