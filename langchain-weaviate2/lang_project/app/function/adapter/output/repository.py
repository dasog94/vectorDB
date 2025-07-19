from datasets import load_dataset
import weaviate

from lang_project.langgraph_search.graph import graph_app
from lang_project.langgraph_search.util.state import GraphState
from lang_project.weaviate_toolkit.WeaviateManager import WeaviateManager
from lang_project.weaviate_toolkit.EmbeddingModelSetup import setup_bge_m3_embeddings
import pandas as pd

class FunctionRepository:
    def __init__(self):
        self.manager = WeaviateManager(
            weaviate_client = weaviate.connect_to_local(),
            embeddings = setup_bge_m3_embeddings()
        )

    def search_hybrid(self, keyword: str, k: int, alpha: float):
        return self.manager.search_hybrid(
            keyword,
            k=k,
            alpha=alpha
        )

    def load_data(self):
        # declare name of the collection
        DATA_SET    = "beomi/KoAlpaca-v1.1a"
        DATA_FILE   = "/Users/jb/Projects/vectorDB/langchain-weaviate2/data/KoAlpaca-train.csv"  # 상대 경로 수정
        TEXT_COLUMN = "instruction"
        INDEX_NAME = "MyDocuments"

        load_data = load_dataset(DATA_SET, split="train")
        df = pd.read_csv(DATA_FILE)
        metadata_df = df.drop(columns=[TEXT_COLUMN])  # 메타데이터 컬럼들

        self.manager.delete_index(INDEX_NAME)

        self.manager.load_dataframe_to_weaviate(
            df=df[:100],
            text_column=TEXT_COLUMN,  # 벡터화할 텍스트 컬럼
            index_name=INDEX_NAME,
            metadata_columns=list(df.columns) # 메타데이터 컬럼들
        )

    def search_langgraph(self, keyword: str) -> GraphState:
        response_dict = graph_app.invoke({"question": keyword, "documents": []})

        return GraphState(**response_dict)