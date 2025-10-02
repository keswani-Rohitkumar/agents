from langgraph.graph import END, StateGraph, START
from graph.graph import *
from langchain.tools import Tool
from langgraph.prebuilt import ToolNode

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("refine_question", refine_question)  # refine_question
workflow.add_node("web_search", web_search)  # get_news
workflow.add_node("summarize_news", summarize_news)   # <-- new step
workflow.add_node("extract_entities", extract_entities)  # extract_entities
workflow.add_node("analyze_stocks", analyze_stocks)  # analyze_stocks
workflow.add_node("generate_report", generate_report)  # web generate_report

# Build graph

workflow.add_edge(START, "refine_question")

workflow.add_edge("refine_question", "web_search")
workflow.add_edge("web_search", "summarize_news")
workflow.add_edge("summarize_news", "extract_entities")
workflow.add_edge("extract_entities", "analyze_stocks")
workflow.add_edge("analyze_stocks", "generate_report")
workflow.add_edge("generate_report", END)

# Compile
app = workflow.compile()