from typing import Any, Dict

from state import GraphState

from langgraphs.ingestion import retriver


def retrieve(state:GraphState)-> Dict[str,Any]:
    print("--RETRIVE--")
    question = state.get("question", "")
    documents=retriver.invoke(question)
    return {"documents":documents,"question":question}


