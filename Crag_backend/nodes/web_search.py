### Search
from langchain_community.tools.tavily_search import TavilySearchResults

def tavily_web_search(question):
    web_search_tool = TavilySearchResults(k=3)
    search_results = web_search_tool.invoke(question)
    return search_results
