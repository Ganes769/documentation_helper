from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

llm=ChatGroq(model="openai/gpt-oss-20b",temperature=0)
class GradeDocument(BaseModel):
    """Binary socre for the Relevance check to question yes or no"""
    binary_score:str=Field(description="Documents are relevent to question yes or no?")

structured_llm_grader=llm.with_structured_output(GradeDocument)
system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader