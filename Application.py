import os
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import ChatOpenAI
from langchain.text_splitter import TokenTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from typing import List

graph = Neo4jGraph()
# environment variables for Neo4j database
os.environ["NEO4J_URI"] = "neo4j+s://ec158e18.databases.neo4j.io"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "3Jp86uGmaEho7iiAcw1kE3JFRI5pwhqviKFGEeQbFJY"

# this is for the llama3.1 model running locally at port number 11434
os.environ["OPENAI_API_KEY"] = "NA"
model = ChatOpenAI(model = "crewai-llama3.1:latest", base_url="http://localhost:11434")
llm_transformer = LLMGraphTransformer(llm=model)
graph_documents = llm_transformer.convert_to_graph_documents(<document>) # whatever information is obtained from the user (this would contain information about the user actually)
# to store <document> in the graph databse, we can simply store the graph as 
graph.add_graph_documents(graph_documents=graph_documents, baseEntityLabel=True, include_source=True)


vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(),
    search_type = "hybrid",
    node_label = "document",
    text_node_properties = ["text"],
    embedding_node_property = "embedding"
)

class Entities(BaseModel):
    "Identifying information about entities"
    names: List[str] = Field(
        ...,
        descripton="All the person, organization, or business entities that "
        "appear in the text",
    )

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are extracting organization and person entities from the text",
        ),
        (
            "human", 
            "Use the given format to extract information from the following "
            "input: {question}"
        ),
    ]
)

entity_chain = prompt | model.with_structured_output(Entities)
print(entity_chain.invoke({"question": "Where was Amelia Earhart born?"}).names)



