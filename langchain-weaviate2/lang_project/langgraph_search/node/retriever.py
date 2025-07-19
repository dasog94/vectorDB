# declare name of the collection
from typing import Dict

import pandas as pd
import weaviate

from lang_project.langgraph_search.util.state import GraphState
from lang_project.weaviate_toolkit.WeaviateManager import WeaviateManager, setup_bge_m3_embeddings

DATA_FILE = "/Users/jb/Projects/vectorDB/langchain-weaviate2/data/KoAlpaca-train.csv"  # 상대 경로 수정
TEXT_COLUMN = "instruction"
INDEX_NAME = "MyDocuments"

manager = WeaviateManager(
            weaviate_client = weaviate.connect_to_local(),
            embeddings = setup_bge_m3_embeddings()
        )

# load_data = load_dataset(DATA_SET, split="train")
df = pd.read_csv(DATA_FILE)
metadata_df = df.drop(columns=[TEXT_COLUMN])  # 메타데이터 컬럼들

manager.delete_index(INDEX_NAME)
vector_store = manager.load_dataframe_to_weaviate(
    df=df[:100],
    text_column=TEXT_COLUMN,  # 벡터화할 텍스트 컬럼
    index_name=INDEX_NAME,
    metadata_columns=list(df.columns)  # 메타데이터 컬럼들
)
retriever = vector_store.as_retriever()

def retrieve(state: GraphState) -> Dict:
    docs = retriever.invoke(state.question)
    return {"question": state.question, "documents": docs}