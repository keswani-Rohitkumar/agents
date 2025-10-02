from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from newsapi import NewsApiClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
os.environ["LANGSMITH_TRACING"] = "true"
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Initialize News API client
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

@tool
def fetch_news(query: str) -> str:
    """Fetch top 5 relevant English news articles for a given query from NewsAPI."""
    try:
        # Query NewsAPI
        articles_response = newsapi.get_everything(
            q=query,
            language="en",
            sort_by="relevancy",
            page_size=5
        )

        articles = articles_response.get("articles", [])

        if not articles:
            return "No relevant news articles found."

        # Format output for LLM context
        formatted_context = []
        for i, article in enumerate(articles):
            title = article.get("title", "No Title")
            description = article.get("description", "No Description")
            source = article.get("source", {}).get("name", "Unknown Source")

            formatted_context.append(
                f"Article {i+1} from {source}:\n"
                f"Title: {title}\n"
                f"Description: {description}\n---"
            )

        return "\n".join(formatted_context)

    except Exception as e:
        return f"Error: Failed to fetch news articles due to an API issue: {e}"

# Wrap into a LangGraph ToolNode
news_tool_node = ToolNode([fetch_news])