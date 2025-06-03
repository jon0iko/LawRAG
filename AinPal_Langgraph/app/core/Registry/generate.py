from neo4j import GraphDatabase

NEO4J_URI = "neo4j://localhost:7999"  
NEO4J_USER = "neo4j"                
NEO4J_PASSWORD = "1917@2024"    

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
            raise

    def close(self):
        if self.__driver is not None:
            self.__driver.close()
            print("Neo4j connection closed.")

    def query(self, query, parameters=None, db=None):
        if self.__driver is None:
            print("Driver not initialized.")
            return None
        
        session = None
        response = None
        try:
            session = self.__driver.session(database=db) if db is not None else self.__driver.session()
            response = list(session.run(query, parameters))
        except Exception as e:
            print(f"Query failed: {e}")
            return None
        finally:
            if session is not None:
                session.close()
        return response

def format_registry_for_prompt(registry_data: list[dict]) -> str:
    """
    Formats the law registry data into a string for LLM prompts.
    Example:
    law_id: 1, title: Contracts Law.
    law_id: 2, title: Something.
    """
    formatted_lines = []
    for item in registry_data:
        formatted_lines.append(f"law_id: {item['law_id']}, title: {item['title']}.")
    return "\n".join(formatted_lines)

def generate_law_registry_list(db_connection: Neo4jConnection) -> list[dict]:
    """
    Fetches all Law nodes from Neo4j and returns a list of their
    law_id and title, sorted by formatted_date (oldest to newest).
    Laws with a null formatted_date are placed at the end.
    """
    cypher_query = """
    MATCH (l:Law)
    WHERE l.law_id IS NOT NULL AND l.title IS NOT NULL
    RETURN l.law_id AS law_id, l.title AS title, l.formatted_date AS formatted_date
    ORDER BY l.formatted_date ASC // Order by formatted_date oldest to newest, NULLs will be at the end
    """
    
    results = db_connection.query(cypher_query)
    
    law_registry = []
    if results:
        for record in results:
            law_registry.append({"law_id": record["law_id"], "title": record["title"]})
    else:
        print("No laws found or query failed.")
        
    return law_registry


if __name__ == "__main__":
    print("Attempting to generate law registry...")
    connection = None
    try:
        connection = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        
        print("\nFetching law data from Neo4j...")
        registry_data = generate_law_registry_list(connection)
        
        if registry_data:
            print(f"\nSuccessfully fetched {len(registry_data)} laws.")
            
            # print("\n--- Law Registry (Raw Data) ---")
            # for item in registry_data:
            #     print(item)
                
            # print("\\n--- Formatted for LLM Prompt ---")
            prompt_string = format_registry_for_prompt(registry_data)
            # print(prompt_string)
            
            with open("law_registry.txt", "w", encoding="utf-8") as f:
                f.write(prompt_string)
            print("\nFormatted registry also saved to law_registry_prompt.txt")
            
        else:
            print("Could not generate registry data.")
            
    except Exception as e:
        print(f"An error occurred during registry generation: {e}")
    finally:
        if connection:
            connection.close()
    print("\nRegistry generation process finished.")