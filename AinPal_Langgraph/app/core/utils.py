import os
from neo4j import GraphDatabase

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

def load_file_context(file_path: str) -> str:
    """Loads content from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return f"Error: File not found at {file_path}"
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return f"Error loading file {file_path}: {e}"

def format_law_registry_for_prompt(registry_content: str) -> str:
    """Formats the law registry for the LLM prompt."""
    if not registry_content or registry_content.startswith("Error:") :
        return "No laws found in the registry or registry file not found."
    
    prompt_lines = ["Here is a list of laws I have access to:"]
    for line in registry_content.strip().split('\\n'):
        if line.startswith("//") or not line.strip(): # Skip comments or empty lines
            continue
        try:
            # Assuming format "law_id: 123, title: The Law Title."
            id_part, title_part = line.split(", title: ", 1)
            law_id = id_part.replace("law_id:", "").strip()
            title = title_part.strip().rstrip('.') # Remove trailing period if any
            prompt_lines.append(f"- {title} (law_id: {law_id})")
        except ValueError:
            # print(f"Skipping malformed line in registry: {line}") # Optional: for debugging
            if line.strip(): # Add non-empty, unparsable lines as is, if any
                 prompt_lines.append(f"- {line.strip()}")
    
    if len(prompt_lines) == 1: # Only the header was added
        return "No valid law entries found in the registry content."
        
    return "\\n".join(prompt_lines)
