import pandas as pd
import weaviate
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain.schema import Document
from typing import List, Optional

from weaviate.client import WeaviateClient


class WeaviateManager:
    def __init__(self, weaviate_client, embeddings):
        """
        Weaviate DataFrame 로더 초기화

        Args:
            weaviate_client: Weaviate 클라이언트 인스턴스
            embeddings: 임베딩 모델 (예: HuggingFaceEmbeddings with BGE-M3)
        """
        self.client: WeaviateClient = weaviate_client
        self.embeddings = embeddings
        self.vector_store: WeaviateVectorStore = Optional[None]

    def close(self):
        """Close the Weaviate client connection."""
        if self.client:
            self.client.close()

    def dataframe_to_documents(self,
                               df: pd.DataFrame,
                               text_column: str,
                               metadata_columns: List[str] = None) -> List[Document]:
        """
        DataFrame을 Langchain Document 객체로 변환

        Args:
            df: 변환할 DataFrame
            text_column: 텍스트 내용이 있는 컬럼명
            metadata_columns: 메타데이터로 사용할 컬럼명 리스트

        Returns:
            List[Document]: Document 객체 리스트
        """
        documents = []

        for idx, row in df.iterrows():
            # 텍스트 내용 추출
            page_content = str(row[text_column])

            # 메타데이터 구성
            metadata = {"row_id": idx}

            if metadata_columns:
                for col in metadata_columns:
                    if col in df.columns:
                        metadata[col] = row[col]
            else:
                # 텍스트 컬럼을 제외한 모든 컬럼을 메타데이터로 사용
                for col in df.columns:
                    if col != text_column:
                        metadata[col] = row[col]

            # Document 객체 생성
            doc = Document(
                page_content=page_content,
                metadata=metadata
            )
            documents.append(doc)

        return documents

    def load_dataframe_to_weaviate(self,
                                   df: pd.DataFrame,
                                   text_column: str,
                                   index_name: str = "Documents",
                                   metadata_columns: List[str] = None,
                                   batch_size: int = 100) -> WeaviateVectorStore:
        """
        DataFrame을 Weaviate에 적재

        Args:
            df: 적재할 DataFrame
            text_column: 벡터화할 텍스트 컬럼명
            index_name: Weaviate 인덱스명
            metadata_columns: 메타데이터로 사용할 컬럼 리스트
            batch_size: 배치 크기

        Returns:
            WeaviateVectorStore: 생성된 벡터 스토어
        """
        print(f"DataFrame을 Document로 변환 중... (총 {len(df)}개 행)")

        # DataFrame을 Document로 변환
        documents = self.dataframe_to_documents(df, text_column, metadata_columns)

        print(f"Weaviate에 적재 중... (배치 크기: {batch_size})")

        # 배치 단위로 처리
        all_docs = []
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            all_docs.extend(batch_docs)
            print(f"진행률: {min(i + batch_size, len(documents))}/{len(documents)}")

        # WeaviateVectorStore 생성 및 문서 적재
        self.vector_store = WeaviateVectorStore.from_documents(
            documents=all_docs,
            embedding=self.embeddings,
            client=self.client,
            index_name=index_name
        )

        print("적재 완료!")
        return self.vector_store

    def delete_index(self, index_name: str):
        """
        Delete a Weaviate collection (index) by name.

        Args:
            index_name: The name of the collection to delete.
        """
        if self.client:
            try:
                self.client.collections.delete(index_name)
                print(f"Index '{index_name}' deleted successfully.")
            except Exception as e:
                print(f"Failed to delete index '{index_name}': {e}")

    def search_similar(self, query: str, k: int = 5) -> List[Document]:
        """
        유사도 검색 수행

        Args:
            query: 검색 쿼리
            k: 반환할 문서 수

        Returns:
            List[Document]: 유사한 문서 리스트
        """
        if not self.vector_store:
            raise ValueError("먼저 데이터를 적재해주세요.")

        return self.vector_store.similarity_search(query, k=k)

    def search_hybrid(self, query: str, k: int = 5, alpha: float = 0.7) -> list[tuple[Document, float]]:
        """
        하이브리드 유사도 검색 수행
        Args:
            query: 검색 쿼리
            k: 반환할 문서 수
            alpha: 텍스트 유사도와 메타데이터 유사도의 가중치 (0~1 사이)
        Returns:
            list[tuple[Document, float]]: 유사한 문서와 해당 유사도 점수의 리스트
        """
        if not self.vector_store:
            raise ValueError("먼저 데이터를 적재해주세요.")

        return self.vector_store.similarity_search_with_score(query, k=k, alpha=alpha)


def setup_bge_m3_embeddings():
    """
    BGE-M3 임베딩 모델 설정

    Returns:
        HuggingFaceEmbeddings: BGE-M3 임베딩 모델
    """
    model_kwargs = {
        'device': 'cpu',  # GPU 사용 시 'cuda', CPU 사용 시 'cpu'
        'trust_remote_code': True
    }
    encode_kwargs = {
        'normalize_embeddings': True,  # 임베딩 정규화
        'batch_size': 32  # 배치 크기 조정
    }

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    return embeddings



# 사용 예제
def main():
    print("\nmain 시작")

    # 1. Weaviate 클라이언트 연결
    weaviate_client = weaviate.connect_to_local()

    # 2. BGE-M3 임베딩 모델 초기화
    embeddings = setup_bge_m3_embeddings()

    # 3. 샘플 DataFrame 생성
    sample_data = {
        'title': ['문서 1', '문서 2', '문서 3'],
        'content': [
            '이것은 첫 번째 문서의 내용입니다.',
            '두 번째 문서에는 다른 정보가 포함되어 있습니다.',
            '세 번째 문서는 또 다른 주제를 다룹니다.'
        ],
        'category': ['기술', '비즈니스', '교육'],
        'author': ['김철수', '이영희', '박민수'],
        'date': ['2024-01-01', '2024-01-02', '2024-01-03']
    }
    df = pd.DataFrame(sample_data)

    # 4. 로더 초기화
    manager = WeaviateManager(weaviate_client, embeddings)

    try:
        # 5. DataFrame을 Weaviate에 적재
        manager.load_dataframe_to_weaviate(
            df=df,
            text_column='content',  # 벡터화할 텍스트 컬럼
            index_name='MyDocuments',
            metadata_columns=['title', 'category', 'author', 'date'],  # 메타데이터 컬럼들
            batch_size=32
        )

        # 6. 검색 테스트
        results = manager.search_similar("기술 관련 정보", k=2)

        print("\n검색 결과:")
        for i, doc in enumerate(results, 1):
            print(f"{i}. {doc.page_content}")
            print(f"   metadata: {doc.metadata}")
            print()

        # 7. 하이브리드 검색 테스트
        results = manager.search_hybrid("기술 관련 정보", k=2, alpha=0.7)

        print("\n하이브리드 검색 결과:")
        for i, doc in enumerate(results, 1):
            print(f"{i}. {doc[0].page_content}")
            print(f"   metadata: {doc[0].metadata}")
            print(f"   score: {doc[1]}")
            print()


    finally:
        # 반드시 연결 종료
        manager.close()

if __name__ == "__main__":
    main()