from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
from neo4j import GraphDatabase
import os

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7999")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
ollama_embedder = OllamaEmbeddings(model="bge-m3");
# query = "What is the capital of France?"
# embeddings = ollamaEmbedder.embed_query(query)
# print(embeddings)


class Neo4jConnection:
    def __init__(self, uri, user, password):
        self.__uri = uri
        self.__user = user
        self.__password = password
        self.__driver = None
        try:
            self.__driver = GraphDatabase.driver(self.__uri, auth=(self.__user, self.__password))
            self.__driver.verify_connectivity()
            print("Successfully connected to Neo4j.")
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            # Depending on the application's needs, you might want to raise the exception
            # or handle it by setting the driver to None and letting other parts of the app
            # deal with an unavailable database. For this pipeline, we'll let it raise.
            raise

    def close(self):
        if self.__driver is not None:
            self.__driver.close()
            print("Neo4j connection closed.")

    def query(self, query, parameters=None, db=None):
        if self.__driver is None:
            print("Neo4j driver not initialized.")
            return [] # Return empty list or raise error
        
        try:
            with self.__driver.session(database=db) as session:
                results = session.run(query, parameters)
                return [record.data() for record in results]
        except Exception as e:
            print(f"Neo4j query failed: {e}")
            return [] # Return empty list or raise error
        

neo4j_conn = Neo4jConnection(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD)


query_embedding = ollama_embedder.embed_query("smoking")
vector_search_query = """
CALL db.index.vector.queryNodes('text_chunks', $top_k, $embedding)
YIELD node, score
RETURN node.chunk_text AS text, node.law_title AS law_title, node.section_number AS section, score
ORDER BY score DESC
"""
params = {"embedding": query_embedding, "top_k": 10}
results = neo4j_conn.query(vector_search_query, params)


formatted_texts = []
for record in results:
    text = record.get("text", "")
    law_title = record.get("law_title", "N/A")
    section = record.get("section", "N/A")
    score = record.get("score", 0.0)
    formatted_texts.append(f"Law: {law_title}, Section: {section} (Similarity: {score:.4f})\nText: {text}")
print("\n---\n".join(formatted_texts) if formatted_texts else "No formatted results from vector search.")