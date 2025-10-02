from langchain_tavily import TavilySearch
from langchain.tools import Tool
from dotenv import load_dotenv
import os
# Instantiate once
load_dotenv()
os.environ["LANGSMITH_TRACING"] = "True"
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

ddg_search = TavilySearch(tavily_api_key = TAVILY_API_KEY,
                          max_result = 5,
                          topic = 'general')

def get_search_results(query: str) -> str:
    """Fetches search results from Tavily search."""
    return ddg_search.run(query)

search_tool = Tool(
    name="get_search_results",
    func=get_search_results,
    description="Fetches search results from Internet."
)
