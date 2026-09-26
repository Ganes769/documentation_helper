from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_tavily import TavilySearch
from state import GraphState

load_dotenv()

web_search_tool = TavilySearch(max_results=3)


def web_search(state: GraphState):

    print("-- web search --")

    question = state.get("question", "")
    documents = state.get("documents", [])

    # Search the web
    tavily_results = web_search_tool.invoke({
        "query": question
    })

    # Get the actual search results
    results = tavily_results["results"]

    # Join the content from each result
    joined_tavily_result = "\n".join(
        [result["content"] for result in results]
    )

    print(joined_tavily_result)

    return {
        "question": question,
        "documents": documents,
    }


if __name__ == "__main__":

    result = web_search(
        state={
            "question": "who is pm of nepal?",
            "documents": [],
        }
    )