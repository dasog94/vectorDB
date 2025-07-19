from typing import List

from langchain_core.documents import Document
from pydantic import BaseModel

class GraphState(BaseModel):
    question: str
    documents: List[Document]