from typing import List
import re
from typing_extensions import TypedDict
from langchain.schema import Document
from tools.question_rewriter import question_rewriter
from tools.news_tool import news_tool_node
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from tools.web_search_tool import *
import os
# --- 1. State Definition (Updated for New Flow) ---
from typing import List, Optional, TypedDict
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

llm = ChatOllama(model='llama3.2')


class GraphState(TypedDict, total=False):
    question: str
    rewritten_question: Optional[str]
    tech_summary: Optional[str]
    financial_analysis: Optional[str]
    entities: List[str]
    generation: Optional[str]
    errors: List[str]



# 1. Question Refinement
def refine_question(state: GraphState) -> GraphState:
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant. 
    User asked: "{question}"
    If this question is vague, rewrite it into a clearer, specific one.
    If it's already good, leave it as-is.
    Return only the rewritten question.
    """)
    rewritten = llm.invoke(prompt.format(question=state["question"])).content
    state["rewritten_question"] = rewritten.strip()
    print(f"[REFINE QUESTION] rewritten_question: {state['rewritten_question']}")
    return state

# 2. News Retrieval (tool)

def get_news(state: GraphState) -> GraphState:
    try:
        # Adapt the input shape if tool_node requires "messages"
        articles = news_tool_node.invoke({
            "messages": [
                {"role": "user", "content": state.get("rewritten_question", "")}
            ]
        })
        state["tech_summary"] = articles
    except Exception as e:
        # Ensure errors list exists
        state.setdefault("errors", [])
        state["errors"].append(str(e))
        state["tech_summary"] = ""
    print(articles)
    return state


def summarize_news(state: GraphState) -> GraphState:
    articles = state.get("articles") or state.get("tech_summary") or ""
    if not articles:
        state.setdefault("errors", [])
        state["errors"].append("No articles to summarize")
        state["news_summary"] = ""
        return state

    prompt = f"Summarize the following news articles in a concise way:\n\n{articles}"
    try:
        summary = llm.invoke(prompt).content
    except Exception as e:
        state.setdefault("errors", []).append(str(e))
        summary = ""

    state["news_summary"] = summary
    print(f"[SUMMARIZE NEWS] news_summary length: {len(summary)}")
    return state


# 3. Entity Extraction
def extract_entities(state: GraphState) -> GraphState:
    summary = state.get("news_summary", "")
    if not summary:
        state.setdefault("errors", []).append("No summary available for entity extraction")
        state["entities"] = []
        return state

    prompt = ChatPromptTemplate.from_template("""
    From this news summary, extract company names or stock tickers or given company names use their stock ticker and can use web search tool if required to do that:
    {summary}
    Return as a Python list.
    """)
    try:
        entities_text = llm.invoke(prompt.format(summary=summary)).content
        state["entities"] = eval(entities_text)
    except:
        state["entities"] = []
    
    print(f"[EXTRACT ENTITIES] entities found: {state['entities']}")
    return state

# 4. Financial Analysis (dummy stock tool call)
def analyze_stocks(state: GraphState) -> GraphState:
    if not state["entities"]:
        state["financial_analysis"] = "No entities found for stock analysis."
        return state

    results = []
    for entity in state["entities"]:
        # Placeholder: replace with real stock tool
        results.append(entity)
    state["financial_analysis"] = "\n".join(results)

    print(f"[ANALYZE STOCKS] financial_analysis: {state['financial_analysis']}")
    return state

# 5. Final Report
def generate_report(state: GraphState) -> GraphState:
    tech_summary = state.get("news_summary", "No news summary available.")
    fin_analysis = state.get("financial_analysis", "No financial analysis available.")

    prompt = ChatPromptTemplate.from_template("""
    Summarize findings.

    Tech news:
    {tech}

    Financial analysis:
    {fin}

    Provide a clear final report.
    """)

    try:
        report = llm.invoke(prompt.format(tech=tech_summary, fin=fin_analysis)).content
    except Exception as e:
        state.setdefault("errors", []).append(str(e))
        report = ""

    state["generation"] = report
    return state


def web_search(state):
    print("---WEB SEARCH---")
    question = state.get("rewritten_question") or state["question"]

    # Ensure documents exists
    documents = state.get("documents", [])

    docs = search_tool.run(question)

    if isinstance(docs, list):
        web_results = [
            Document(
                page_content=d.get("title") or d.get("content") or str(d),
                metadata=d,
            )
            for d in docs
        ]
    else:
        web_results = [Document(page_content=str(docs), metadata={"source": "web_search"})]

    documents.extend(web_results)
    state["documents"] = documents

    print(f"[WEB SEARCH] documents in state: {len(state['documents'])}")
    return state    

#return {**state, "documents": documents, "question": question}
