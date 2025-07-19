from typing import Dict, List, Any, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.documents import Document
import json

from .WeaviateManager import WeaviateManager


class SearchState(TypedDict):
    """검색 상태를 관리하는 클래스"""
    query: str
    k: int
    alpha: float
    search_results: List[tuple[Document, float]]
    processed_results: List[Dict[str, Any]]
    error: str


class LangGraphSearchSystem:
    def __init__(self, weaviate_manager: WeaviateManager):
        """
        LangGraph 기반 검색 시스템 초기화

        Args:
            weaviate_manager: WeaviateManager 인스턴스
        """
        self.weaviate_manager = weaviate_manager
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """LangGraph 워크플로우 구축"""

        # 상태 그래프 생성
        workflow = StateGraph(SearchState)

        # 노드 추가
        workflow.add_node("validate_query", self._validate_query)
        workflow.add_node("perform_search", self._perform_search)
        workflow.add_node("process_results", self._process_results)
        workflow.add_node("handle_error", self._handle_error)

        # 엣지 설정
        workflow.set_entry_point("validate_query")

        # 성공 경로
        workflow.add_edge("validate_query", "perform_search")
        workflow.add_edge("perform_search", "process_results")
        workflow.add_edge("process_results", END)

        # 에러 경로
        workflow.add_conditional_edges(
            "validate_query",
            self._should_continue,
            {
                "continue": "perform_search",
                "error": "handle_error"
            }
        )

        workflow.add_conditional_edges(
            "perform_search",
            self._should_continue,
            {
                "continue": "process_results",
                "error": "handle_error"
            }
        )

        workflow.add_edge("handle_error", END)

        return workflow.compile()

    def _validate_query(self, state: SearchState) -> SearchState:
        """쿼리 유효성 검사"""
        query = state.get("query", "").strip()

        if not query:
            state["error"] = "검색 쿼리가 비어있습니다."
            return state

        if len(query) < 2:
            state["error"] = "검색 쿼리는 최소 2자 이상이어야 합니다."
            return state

        # 기본값 설정
        if "k" not in state:
            state["k"] = 5
        if "alpha" not in state:
            state["alpha"] = 0.7

        return state

    def _perform_search(self, state: SearchState) -> SearchState:
        """하이브리드 검색 수행"""
        try:
            query = state["query"]
            k = state["k"]
            alpha = state["alpha"]

            # WeaviateManager의 search_hybrid 메서드 호출
            results = self.weaviate_manager.search_hybrid(
                query=query,
                k=k,
                alpha=alpha
            )

            state["search_results"] = results
            return state

        except Exception as e:
            state["error"] = f"검색 중 오류가 발생했습니다: {str(e)}"
            return state

    def _process_results(self, state: SearchState) -> SearchState:
        """검색 결과 처리 및 포맷팅"""
        try:
            results = state["search_results"]
            processed_results = []

            for i, (doc, score) in enumerate(results, 1):
                processed_result = {
                    "rank": i,
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score),
                    "formatted_score": f"{score:.4f}"
                }
                processed_results.append(processed_result)

            state["processed_results"] = processed_results
            return state

        except Exception as e:
            state["error"] = f"결과 처리 중 오류가 발생했습니다: {str(e)}"
            return state

    def _handle_error(self, state: SearchState) -> SearchState:
        """에러 처리"""
        error = state.get("error", "알 수 없는 오류가 발생했습니다.")
        state["processed_results"] = [{"error": error}]
        return state

    def _should_continue(self, state: SearchState) -> str:
        """다음 단계 결정"""
        if "error" in state and state["error"]:
            return "error"
        return "continue"

    def search(self, query: str, k: int = 5, alpha: float = 0.7) -> Dict[str, Any]:
        """
        LangGraph를 사용한 하이브리드 검색 수행

        Args:
            query: 검색 쿼리
            k: 반환할 문서 수
            alpha: 하이브리드 검색 가중치 (0~1)

        Returns:
            Dict[str, Any]: 검색 결과 및 메타데이터
        """
        initial_state = SearchState(
            query=query,
            k=k,
            alpha=alpha,
            search_results=[],
            processed_results=[],
            error=""
        )

        # 그래프 실행
        result = self.graph.invoke(initial_state)

        return {
            "query": result["query"],
            "k": result["k"],
            "alpha": result["alpha"],
            "results": result["processed_results"],
            "total_results": len(result["processed_results"]),
            "has_error": bool(result.get("error")),
            "error": result.get("error", "")
        }


# LangGraph 툴로 사용하기 위한 래퍼 함수들
@tool
def hybrid_search_tool(query: str, k: int = 5, alpha: float = 0.7) -> str:
    """
    Weaviate 하이브리드 검색을 수행하는 툴

    Args:
        query: 검색할 쿼리 문자열
        k: 반환할 최대 문서 수 (기본값: 5)
        alpha: 하이브리드 검색 가중치 (0~1, 기본값: 0.7)

    Returns:
        str: JSON 형태의 검색 결과
    """
    # 이 함수는 WeaviateManager 인스턴스가 필요하므로
    # 실제 사용 시에는 적절한 컨텍스트에서 호출되어야 합니다
    return f"검색 쿼리: {query}, k: {k}, alpha: {alpha}"


class SearchToolNode:
    """LangGraph에서 사용할 수 있는 검색 툴 노드"""

    def __init__(self, weaviate_manager: WeaviateManager):
        self.weaviate_manager = weaviate_manager
        self.search_system = LangGraphSearchSystem(weaviate_manager)

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """상태에서 검색을 수행하고 결과를 반환"""
        query = state.get("query", "")
        k = state.get("k", 5)
        alpha = state.get("alpha", 0.7)

        search_result = self.search_system.search(query, k, alpha)

        # 상태에 결과 추가
        state["search_results"] = search_result["results"]
        state["search_metadata"] = {
            "total_results": search_result["total_results"],
            "has_error": search_result["has_error"],
            "error": search_result["error"]
        }

        return state


# 사용 예제
def create_search_workflow(weaviate_manager: WeaviateManager) -> StateGraph:
    """
    검색 워크플로우 생성

    Args:
        weaviate_manager: WeaviateManager 인스턴스

    Returns:
        StateGraph: 검색 워크플로우
    """

    # 워크플로우 상태 정의
    class WorkflowState(TypedDict):
        query: str
        k: int
        alpha: float
        search_results: List[Dict[str, Any]]
        search_metadata: Dict[str, Any]
        final_response: str

    # 그래프 생성
    workflow = StateGraph(WorkflowState)

    # 검색 툴 노드 생성
    search_tool = SearchToolNode(weaviate_manager)

    # 노드 추가
    workflow.add_node("search", search_tool)
    workflow.add_node("format_response", lambda state: {
        **state,
        "final_response": json.dumps(state["search_results"], ensure_ascii=False, indent=2)
    })

    # 엣지 설정
    workflow.set_entry_point("search")
    workflow.add_edge("search", "format_response")
    workflow.add_edge("format_response", END)

    return workflow.compile()


# 테스트 함수
def test_langgraph_search(weaviate_manager: WeaviateManager):
    """LangGraph 검색 시스템 테스트"""
    print("LangGraph 검색 시스템 테스트 시작...")

    # 검색 시스템 생성
    search_system = LangGraphSearchSystem(weaviate_manager)

    # 테스트 쿼리들
    test_queries = [
        "기술 관련 정보",
        "비즈니스 문서",
        "교육 자료"
    ]

    for query in test_queries:
        print(f"\n=== 쿼리: {query} ===")
        result = search_system.search(query, k=3, alpha=0.7)

        print(f"총 결과 수: {result['total_results']}")
        if result['has_error']:
            print(f"오류: {result['error']}")
        else:
            for i, doc_result in enumerate(result['results'], 1):
                print(f"{i}. 점수: {doc_result['formatted_score']}")
                print(f"   내용: {doc_result['content'][:100]}...")
                print(f"   메타데이터: {doc_result['metadata']}")
                print()

    print("LangGraph 검색 시스템 테스트 완료!")