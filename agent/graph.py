from groq import Groq
from langgraph.graph import StateGraph, START, END

from .state import BloodTestState
from .nodes import create_categorize_node, create_explain_node, create_summary_node


def create_graph(groq_api_key: str):
    client = Groq(api_key=groq_api_key)

    graph = StateGraph(BloodTestState)

    graph.add_node("categorize_values", create_categorize_node(client))
    graph.add_node("explain_values", create_explain_node(client))
    graph.add_node("generate_summary", create_summary_node(client))

    graph.add_edge(START, "categorize_values")
    graph.add_edge("categorize_values", "explain_values")
    graph.add_edge("explain_values", "generate_summary")
    graph.add_edge("generate_summary", END)

    return graph.compile()
