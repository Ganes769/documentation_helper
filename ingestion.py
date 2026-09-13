import asyncio
import os
import ssl
from typing import Any,Dict,List
import certifi
from dotenv import load_dotenv
from langchain_core.documents import  Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl,TavilyExtract,TavilyMap, tavily_crawl, tavily_map
from numpy import size
from backend.core import vectorstore
from logger import (log_error,log_header,log_info,log_success,log_warning)
load_dotenv()
ssl_context=ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"]=certifi.where()
os.environ["REQUESTS_CA_BUNDLE"]=certifi.where()
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

tavily_crawl = TavilyCrawl()

vectorstore=PineconeVectorStore(index_name="documentation-helper",embedding=embeddings)
tavily_extract=TavilyExtract()
tavily_map=TavilyMap(max_depth=5,max_breadth=20,max_pages=1000)

async def main():
    log_header("Documentation ingestion pipeline")
    # Crawl  the document size
    res=tavily_crawl.invoke({
        "url":"https://docs.langchain.com/",
        "max_depth":5,
        "extract_depth":"advanced" ,
        "instructions":"content in a agent"
        })
    all_docs=[Document(page_content=result["raw_content"],metadata={"source":result["url"]}) for result in res["results"]]
    print("content",all_docs)
    log_success("successfully crawaled")

    # print("res",all_docs.raw_content)


if __name__=="__main__":
 asyncio.run(main())
    