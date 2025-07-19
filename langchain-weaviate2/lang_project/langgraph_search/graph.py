from langgraph.graph import StateGraph

from lang_project.langgraph_search.node.retriever import retrieve
from lang_project.langgraph_search.util.state import GraphState

workflow = StateGraph(GraphState)

# Define the workflow
workflow.add_node("retrieve", retrieve)
# workflow.add_node("generate", generate)

workflow.set_entry_point("retrieve")
workflow.set_finish_point("retrieve") # temp

# workflow.add_edge("retrieve", "generate")
# workflow.add_edge("generate", END)

graph_app = workflow.compile()

graph_app.get_graph().draw_mermaid_png(output_file_path="./graph.png")
