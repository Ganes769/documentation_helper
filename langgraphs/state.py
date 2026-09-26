from typing import TypedDict


class GraphState(TypedDict):
    """
Represtent the state of the graph:
Attributes:
question:question
generation:LLM generation
web_search:Weather to add search
documents:lists of documents

"""

question:str
generation:str
web_search:bool
documents:list[str]
