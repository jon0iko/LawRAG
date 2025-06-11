# Define the maximum number of allowed tool calls
MAX_TOOL_CALLS = 15












# --------------------------------------------------------------------- #
# AUTHOR: jon0iko
# DATE: 03-06-2025
# --------------------------------------------------------------------- #
# --------------------------------------------------------------------- #
# CODE STARTS HERE
# --------------------------------------------------------------------- #
import os
import sys
import json
import tiktoken
from dotenv import load_dotenv

# Token counting utilities
def count_tokens(text, model="gpt-4"):
    """Count the number of tokens in a text string."""
    if not text:
        return 0
    
    # Select encoding based on model
    if model.startswith("gpt-4"):
        encoding = tiktoken.encoding_for_model("gpt-4")
    elif model.startswith("gpt-3.5"):
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    else:
        encoding = tiktoken.get_encoding("cl100k_base")  # Default for newer models
    
    return len(encoding.encode(text))

# Global token counters
input_tokens = 0
output_tokens = 0

if __name__ == "__main__" and __package__ is None:
    current_script_path = os.path.abspath(__file__)
    core_dir = os.path.dirname(current_script_path)
    app_dir = os.path.dirname(core_dir)
    project_root_for_imports = os.path.dirname(app_dir)
    if project_root_for_imports not in sys.path:
        sys.path.insert(0, project_root_for_imports)
from typing import TypedDict, Annotated, Sequence, Literal, Optional, Any, List, Dict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool 
from langchain_core.utils.utils import convert_to_secret_str
from langchain_ollama import OllamaEmbeddings
from langgraph.graph import StateGraph, END
import operator
from openai import OpenAI
from langchain_openai import ChatOpenAI
from app.core.utils import Neo4jConnection, load_file_context, format_law_registry_for_prompt
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.sqlite import SqliteSaver


load_dotenv()


# memory for persistence
# memory = AsyncSqliteSaver.from_conn_string(":memory:")
# memory = SqliteSaver.from_conn_string("sqlite:///path/to/your/database.db")

# --- Constants & Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = "bge-m3"

MAX_RETRIES_REASONING = 1  # Max number of retries for the reasoning agent

openai_client: Optional[OpenAI] = None
if OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print("OpenAI client initialized.")
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
else:
    print("OPENAI_API_KEY not found. OpenAI client will not be available.")

neo4j_conn: Optional[Neo4jConnection] = None
if NEO4J_USER and NEO4J_PASSWORD:  # Basic check for credentials
    try:
        neo4j_conn = Neo4jConnection(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD)
        print("Neo4j connection object created.")
        # Test connection by running a simple query
        neo4j_conn.query("MATCH (n) RETURN n LIMIT 1")
        print("Successfully connected to Neo4j database.")
    except Exception as e:
        print(f"Failed to initialize Neo4j connection object: {e}")
        neo4j_conn = None
else:
    print("Neo4j credentials (NEO4J_USER, NEO4J_PASSWORD) not found. Neo4j client will not be available.")

ollama_embedder: Optional[OllamaEmbeddings] = None
try:
    # Test Ollama availability by trying to initialize
    ollama_embedder = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    ollama_embedder.embed_query("test")
    print(f"Ollama embedder initialized with model {OLLAMA_EMBED_MODEL} at {OLLAMA_BASE_URL}.")
except Exception as e:
    print(
        f"Failed to initialize Ollama embedder or test embedding: {e}. Ensure Ollama is running, model '{OLLAMA_EMBED_MODEL}' is available, and {OLLAMA_BASE_URL} is correct.")
    ollama_embedder = None

# Construct absolute paths to the files
CORE_DIR = os.path.dirname(os.path.abspath(__file__))
LAW_REGISTRY_PATH = os.path.join(CORE_DIR, "law_registry.txt")
DB_SCHEMA_PATH = os.path.join(CORE_DIR, "databaseschema.txt")

law_registry_content = load_file_context(LAW_REGISTRY_PATH)
db_schema_content = load_file_context(DB_SCHEMA_PATH)
formatted_law_registry_prompt = format_law_registry_for_prompt(law_registry_content)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    retrieved_law_texts: List[str]
    evaluation_result: Optional[Literal["sufficient", "insufficient"]]
    reasoning_retries_count: int
    final_answer: Optional[str]
    error_message: Optional[str]
    cypher_query_results: Optional[List[Dict[str, Any]]]
    tool_calls_count: int  # Track number of tool calls
    force_evaluation: bool  # Flag to force evaluation

@tool
def hybrid_search_tool(expanded_query: str) -> dict:
    """
    This tool performs a hybrid search through the neo4j law database
    and retrieves relevant law texts based on the expanded query.
    It will do both keyword search and vector search using Ollama embeddings.
    Make sure to use such an expanded query using English and/or Bangla that retrieves the best possible results.
    """
    if ollama_embedder is None:
        return {"error_message": "Ollama embedder is not available. Please check the configuration."}
    
    if neo4j_conn is None:
        return {"error_message": "Neo4j connection is not available. Please check the configuration."}
    
    try:
        # Vector search
        queryVector = ollama_embedder.embed_query(expanded_query)
        vectorsearchquery = """
        CALL db.index.vector.queryNodes('text_chunks', 8, $vector)
        YIELD node, score
        RETURN node, score
        """
        vectorResults = neo4j_conn.query(vectorsearchquery, parameters={"vector": queryVector})
        
        # Keyword search
        keywords = [word for word in expanded_query.lower()
                   .replace(r'[^\w\s]', '')
                   .split()
                   if len(word) > 3]
        
        keywordResults = []
        if keywords:
            keywordSearchQuery = """
            MATCH (chunk:Chunk)
            WHERE """ + " OR ".join([f"toLower(chunk.chunk_text) CONTAINS $keyword{i}" for i in range(len(keywords))]) + """
            RETURN chunk as node, 0.5 as score
            LIMIT 8
            """
            
            keywordParams = {f"keyword{i}": keyword for i, keyword in enumerate(keywords)}
            keywordResults = neo4j_conn.query(keywordSearchQuery, parameters=keywordParams)
        
        # Combine results
        allResults = vectorResults + keywordResults
        uniqueNodes = {}
        
        for record in allResults:
            node = record.get("node")
            score = record.get("score")
            nodeId = str(node.get("chunk_id", ""))
            
            if nodeId not in uniqueNodes or uniqueNodes[nodeId]["score"] < score:
                uniqueNodes[nodeId] = {"node": node, "score": score}
        
        # Sort and limit combined results
        combinedResults = sorted(uniqueNodes.values(), key=lambda x: x["score"], reverse=True)[:10]
        
        # Expand to adjacent chunks
        expandedNodes = set()
        expandedResults = []
        
        for result in combinedResults:
            node = result["node"]
            nodeId = str(node.get("chunk_id", ""))
            
            if nodeId not in expandedNodes:
                expandedNodes.add(nodeId)
                expandedResults.append(result)
                
                # Fetch adjacent chunks
                expandQuery = """
                MATCH (node)-[:NEXT_CHUNK*1..2]->(next_chunk)
                WHERE node.chunk_id = $nodeId
                RETURN next_chunk as node, 0.4 as score
                LIMIT 3
                """
                
                expandedChunks = neo4j_conn.query(expandQuery, parameters={"nodeId": nodeId})
                
                for expandedRecord in expandedChunks:
                    expandedNode = expandedRecord.get("node")
                    expandedNodeId = str(expandedNode.get("chunk_id", ""))
                    
                    if expandedNodeId not in expandedNodes:
                        expandedNodes.add(expandedNodeId)
                        expandedResults.append({
                            "node": expandedNode,
                            "score": expandedRecord.get("score")
                        })
        
        # Format results
        formatted_results = []
        retrieved_law_texts = []
        
        for result in expandedResults:
            node = result["node"]
            score = result["score"]
            chunk_text = node.get("chunk_text", "")
            law_title = node.get("law_title", "")
            section_number = node.get("section_number", "")
            
            formatted_text = f"Law: {law_title}, Section: {section_number} (Similarity: {score:.4f})\nText: {chunk_text}"
            formatted_results.append(formatted_text)
            
            # Add to retrieved_law_texts for AgentState
            retrieved_law_texts.append(formatted_text)
            
            print(f"[Score: {score:.3f}] [{law_title} | Section {section_number}] {chunk_text}")
            print("-------------------------------------")
        
        # Return properly formatted AgentState with retrieved documents
        return {
            "retrieved_law_texts": retrieved_law_texts
        }
        
    except Exception as e:
        error_msg = f"Error during hybrid search: {e}"
        print(error_msg)
        return {"error_message": error_msg}
        


@tool
def cypher_query_runner_tool(cypher_query: str) -> dict:
    """
    This tool runs a Cypher query against the Neo4j database.
    It returns the results of the query or an error message if it fails.
    """
    if neo4j_conn is None:
        return {"error_message": "Neo4j connection is not available. Please check the configuration."}
    
    try:
        results = neo4j_conn.query(cypher_query)
        if not results:
            return {"cypher_query_results": [], "error_message": "No results found for the query."}
        
        return {"cypher_query_results": results, "error_message": None}
    
    except Exception as e:
        error_msg = f"Error running Cypher query: {e}"
        print(error_msg)
        return {"cypher_query_results": [], "error_message": error_msg}
    
tools = [hybrid_search_tool, cypher_query_runner_tool]


OPENAI_API_KEY_SECRETSTR = convert_to_secret_str(OPENAI_API_KEY) if OPENAI_API_KEY else None


reasoning_llm = ChatOpenAI(
    model="o4-mini-2025-04-16",
    api_key=OPENAI_API_KEY_SECRETSTR,
    max_retries=MAX_RETRIES_REASONING,
)

reasoning_llm = reasoning_llm.bind_tools(tools)

def reasoning_agent_node(state: AgentState) -> AgentState:
    """
    This node is solely responsible for reasoning how to answer the user's query by utilizing the available tools.
    It will first see the user's query and decide if it is a law related query or not.
    If it is not related to law, it will return an error message.
    If it is a greeting, it will return a greeting message.
    if it a law related query, the reasoning agent llm will reason how to approach the query.
    The law_registry_content and db_schema_content are provided to the LLM as context. It will use this information to deccide which tool or tools to use.
    If it decides to use the hybrid_search_tool, it will call the tool with a properly expanded query.
    If it decides to use the cypher_query_runner_tool, it will call the tool with a properly formatted Cypher query.
    The reasoning agent will also handle retries if the LLM fails to provide a valid response.
    """

    if not state['messages']:
        raise ValueError("State must contain at least one message.")

    messages = state['messages']

    last_message = messages[-1]
    
    print(f"Last message content: {last_message.content}")

    user_query = str(last_message.content).strip()
    state['retrieved_law_texts'] = []
    state['evaluation_result'] = None
    state['reasoning_retries_count'] = 0
    state['final_answer'] = None
    state['error_message'] = None
    state['cypher_query_results'] = None
    state['tool_calls_count'] = 0  # Initialize the tool calls count
    state['force_evaluation'] = False  # Initialize the force evaluation flag
    
    # Prepare the system prompt with context
    system_prompt = f"""
    Suppose, you are an intelligent AI assistant who can think and reason like a lawyer practicing in Bangladesh.
    Your only job is to process a user's query and retrieve the most relevant law sections from the a law database through the usage of two tools.
    Use your reasoning skills to determine the best approach to answer the user's query. 
    You have two tools: hybrid_search_tool and cypher_query_runner_tool. Prioritize the hybrid_search_tool for retrieving law sections based on semantic similarity search, and use the cypher_query_runner_tool for executing specific Cypher queries against the Neo4j database.
    The hybrid_search_tool allows you to perform a multilingual semantic search using an expanded query that includes relevant keywords. The database contains laws in both English and Bangla, so you can use either language in your queries.
    There is a law registry given below. The law titles in the law_registry that are in Bangla have the section_number field and law texts also in Bangla, I mean in Bangla fonts. So, use the tools accordingly. Include Bangla words in the expanded query if need be.
    The cypher_query_runner_tool allows you to run specific Cypher queries against the Neo4j database to retrieve metadata or specific sections of laws.
    You can use the hybrid_search_tool to retrieve relevant law sections based on the user's query, and then use the cypher_query_runner_tool to extract specific metadata or sections from the retrieved results.
    You have access to a law database of all Bangladeshi laws that has these laws stored in a formatted manner.
    The id and title of laws in the database are as follows:
    {formatted_law_registry_prompt}
    The database schema of the Neo4j database is as follows:
    {db_schema_content}
    You can make multiple calls to the tools if needed.
    If the query is not related to law, return an error message saying "This query is not related to law."
    If the query is a greeting, return a greeting message.
    If the query is a law related query, reason how to approach the query and then use the tool/s.
    If you need to look up some information before asking a follow up question, you are allowed to do that!
    Keep in mind while generating the cypher query, that if the law title is in Bangla in the law registry, then the section_number field in the database will also be in Bangla.
    If the law title is in English, then the section_number field will be in English.
    IF you think enough information has been retrieved, you can stop calling the tools and return an AIMessage.
    """

    # if messages does not contain a SystemMessage, add one
    if not any(isinstance(msg, SystemMessage) for msg in messages):
        messages = [SystemMessage(content=system_prompt)] + list(messages)

    # Track input tokens
    global input_tokens
    for msg in messages:
        input_tokens += count_tokens(str(msg.content), "gpt-4")

    # state['reasoning_retries_count'] += 1
    # if state['reasoning_retries_count'] > MAX_RETRIES_REASONING:
    #     state['error_message'] = "Reasoning agent has exceeded maximum retries."
    #     return state

    answer = reasoning_llm.invoke(messages)
    print(f"Reasoning LLM Response: {answer.content}")
    
    # Track output tokens
    global output_tokens
    output_tokens += count_tokens(answer.content, "gpt-4")

    return {
        'messages': [answer],
        'retrieved_law_texts': state['retrieved_law_texts'],
        'evaluation_result': state['evaluation_result'],
        'reasoning_retries_count': state['reasoning_retries_count'],
        'final_answer': state['final_answer'],
        'error_message': state['error_message'],
        'cypher_query_results': state['cypher_query_results']
    }

tools_dict = {tool.name: tool for tool in tools}

def execute_tools_node(state: AgentState) -> AgentState:
    """
    This node executes the tools based on the reasoning agent's response.
    It will check the last message for tool calls and execute them, then add ToolMessage responses.
    It also tracks the number of tool calls and limits them to MAX_TOOL_CALLS.
    """
    
    
    # Get tool_calls_count from state or initialize it
    tool_calls_count = state.get('tool_calls_count', 0)
    
    if not state['messages']:
        raise ValueError("State must contain at least one message.")

    last_message = state['messages'][-1]
    
    if not isinstance(last_message, AIMessage):
        raise ValueError("Last message must be an AIMessage for tool execution.")

    tool_calls = getattr(last_message, 'tool_calls', [])

    if not tool_calls:
        return state
    
    # Increment the tool calls count
    tool_calls_count += 1
    print(f"Tool call {tool_calls_count} of {MAX_TOOL_CALLS}")
    
    # Check if we've exceeded the maximum number of tool calls
    if tool_calls_count >= MAX_TOOL_CALLS:
        print(f"Reached maximum number of tool calls ({MAX_TOOL_CALLS}). Moving to evaluation.")
        
        # Add a message to inform about the limit
        forced_stop_message = ToolMessage(
            content=f"Maximum number of tool calls ({MAX_TOOL_CALLS}) reached. Proceeding to final answer.",
            tool_call_id="max_calls_reached",
            name="system"
        )
        
        # Force the next step to be evaluation by adding a special flag
        return {
            'messages': [forced_stop_message],
            'retrieved_law_texts': state['retrieved_law_texts'],
            'cypher_query_results': state['cypher_query_results'],
            'error_message': None,
            'evaluation_result': "sufficient",  # Force evaluation
            'reasoning_retries_count': state['reasoning_retries_count'],
            'final_answer': state['final_answer'],
            'tool_calls_count': tool_calls_count,
            'force_evaluation': True  # Add this flag to force evaluation
        }

    results = []
    tool_messages = []
    
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]
        
        print(f"Executing tool: {tool_name} with args: {tool_args}")
        
        if tool_name in tools_dict:
            tool = tools_dict[tool_name]
            try:
                result = tool.invoke(tool_args)
            except Exception as e:
                error_msg = f"Error executing tool {tool_name}: {str(e)}"
                print(error_msg)
                result = {"error_message": error_msg}
        else:
            error_msg = f"Unknown tool called: {tool_name}"
            print(error_msg)
            result = {"error_message": error_msg}

        # Create a ToolMessage for each tool call
        if isinstance(result, dict):
            # Convert dict to string for the message content
            if "error_message" in result and result["error_message"]:
                content = f"Error: {result['error_message']}"
            elif "retrieved_law_texts" in result and result["retrieved_law_texts"]:
                content = f"Found {len(result['retrieved_law_texts'])} relevant law texts:\n\n" + "\n\n".join(result["retrieved_law_texts"][:3])
                if len(result["retrieved_law_texts"]) > 3:
                    content += "\n..."
            elif "cypher_query_results" in result and result["cypher_query_results"]:
                content = f"Query results: {json.dumps(result['cypher_query_results'][:5], indent=2)}"
            else:
                content = json.dumps(result, indent=2)
        else:
            content = str(result)
        
        # Create the tool message
        tool_message = ToolMessage(
            content=content,
            tool_call_id=tool_call_id,
            name=tool_name
        )
        
        tool_messages.append(tool_message)
        results.append(result)

    # Process results and update state
    retrieved_law_texts = []
    cypher_query_results = None
    error_message = None
    
    for result in results:
        if isinstance(result, dict):
            if 'retrieved_law_texts' in result:
                retrieved_law_texts.extend(result['retrieved_law_texts'])
            if 'cypher_query_results' in result:
                cypher_query_results = result['cypher_query_results']
            if 'error_message' in result and result['error_message']:
                error_message = result['error_message']

    # Add the tool messages to the conversation history
    return {
        'messages': tool_messages,
        'retrieved_law_texts': retrieved_law_texts,
        'cypher_query_results': cypher_query_results,
        'error_message': error_message,
        'evaluation_result': state['evaluation_result'],
        'reasoning_retries_count': state['reasoning_retries_count'],
        'final_answer': state['final_answer'],
        'tool_calls_count': tool_calls_count  # Store the updated count
    }
    

def reasoning_should_continue(state: AgentState) -> bool:
    """
    This node checks if the reasoning agent should continue or stop.
    It will return True if the reasoning agent should continue, otherwise False.
    """
    # Check if we need to force evaluation due to max tool calls
    if state.get('force_evaluation', False):
        print("Forcing evaluation due to maximum tool calls reached")
        return False
    
    if not state['messages']:
        raise ValueError("State must contain at least one message.")

    last_message = state['messages'][-1]
    
    if not isinstance(last_message, AIMessage):
        raise ValueError("Last message must be an AIMessage.")

    # Check if the last message contains tool calls
    return len(last_message.tool_calls) > 0 and hasattr(last_message, 'tool_calls')


def combine_results_and_evaluate_node(state: AgentState) -> AgentState:
    """
    This node combines the results from the tools and evaluates if the answer is sufficient.
    It will set the evaluation_result to "sufficient" or "insufficient" based on the retrieved texts.
    """
    if not state['retrieved_law_texts'] and not state['cypher_query_results']:
        state['evaluation_result'] = "insufficient"
        state['final_answer'] = "No relevant laws found."
        return state
    
    last_ai_message = None
    for msg in reversed(state['messages']):
        if isinstance(msg, AIMessage) and not getattr(msg, 'tool_calls', None):
            last_ai_message = msg
            break

    # Combine retrieved texts into a final answer
    if state['cypher_query_results']:
        cypher_results_texts = [f"Cypher Query Result: {json.dumps(result)}" for result in state['cypher_query_results']]
        combined_text = "\n".join(state['retrieved_law_texts'] + cypher_results_texts)
    else:
        combined_text = "\n".join(state['retrieved_law_texts'])

    if last_ai_message:
        combined_text = f"Reasoning Agent's Analysis:\n{last_ai_message.content}\n\nRetrieved Law Texts:\n{combined_text}"

    print(f"Combined Text for Final Answer:\n{combined_text}")

    # Set final answer and evaluation result
    state['final_answer'] = combined_text
    state['evaluation_result'] = "sufficient"

    return state

# def greeting_node(state: AgentState) -> AgentState:
#     """
#     This node checks if the user's query is a greeting
#     If it is a greeting, it will return a greeting message.
#     """
#     if not state['messages']:
#         raise ValueError("State must contain at least one message.")

#     last_message = state['messages'][-1]
    
#     if not isinstance(last_message, HumanMessage):
#         raise ValueError("Last message must be a HumanMessage.")

#     user_query = str(last_message.content).strip().lower()

#     # Check for greetings
#     greetings = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"]
    
#     if any(greeting in user_query for greeting in greetings):
#         state['final_answer'] = "Hello! How can I assist you with your legal query today?"
#         state['evaluation_result'] = "sufficient"
#         return state



answerer_llm = ChatOpenAI(
    model="gpt-4.1-mini-2025-04-14",
    temperature=0.0,
    api_key=OPENAI_API_KEY_SECRETSTR,
)

def answerer_agent_node(state: AgentState) -> AgentState:
    """
    This node is responsible for generating the final answer based on the retrieved law texts and evaluation result.
    It will use the answerer_llm to generate a concise answer.
    """
    if not state['final_answer']:
        state['error_message'] = "No final answer available."
        return state

    messages = [
        SystemMessage(content="You are an AI assistant that provides concise answers based on retrieved law texts only. Don't use knowledge outside this. Quote the law sections in your answer. Explain everything in layman's terms. If the information is not enough to answer the query, just say 'I don't have enough information to answer your question."),
        HumanMessage(content=state['final_answer'])
    ]

    # Track input tokens
    global input_tokens
    for msg in messages:
        input_tokens += count_tokens(str(msg.content), "gpt-4")

    answer = answerer_llm.invoke(messages)
    
    # Track output tokens
    global output_tokens
    output_tokens += count_tokens(answer.content, "gpt-4")

    state['messages'] = list(state['messages']) + [answer]
    return state


graph = StateGraph(AgentState)
# graph.add_node("greeting_or_irrelevant_query", greeting_or_irrelevant_query_node)
# graph.add_conditional_edges("greeting_or_irrelevant_query", lambda state: state['evaluation_result'] == "sufficient", {False: "reasoning_agent", True: END})
graph.add_node("reasoning_agent", reasoning_agent_node)
graph.add_node("execute_tools", execute_tools_node)
graph.add_conditional_edges("reasoning_agent", reasoning_should_continue, {True: "execute_tools", False: "combine_results_and_evaluate"})
graph.add_node("combine_results_and_evaluate", combine_results_and_evaluate_node)
graph.add_node("answerer_agent", answerer_agent_node)
graph.add_conditional_edges("combine_results_and_evaluate", lambda state: state['evaluation_result'] == "sufficient", {True: "answerer_agent", False: END})
graph.add_edge("execute_tools", "reasoning_agent")
graph.set_entry_point("reasoning_agent")

# db_path = os.path.join(os.path.dirname(__file__), "temp_graph_checkpoint.db")

# app = graph.compile()
# graph_image_path = os.path.join(os.path.dirname(__file__), "graph_structure.png")
# png_data = app.get_graph().draw_mermaid_png()
# # Save the graph structure to a file
# with open(graph_image_path, "wb") as f:
#     f.write(png_data)

with SqliteSaver.from_conn_string(":memory:") as checkpointer:
    app = graph.compile(checkpointer=checkpointer)
    thread = {"configurable": {"thread_id": "5"}}
    conversation_history = []

    while True:
        # Get user input
        query = input("\nEnter your query (or type 'exit' to quit): ").strip()
        if query.lower() in ['exit', 'quit', 'bye', 'goodbye']:
            print("Exiting")
            break
        new_message = HumanMessage(content=query)
        full_messages = conversation_history + [new_message]

        print("\n=== Starting Graph Execution ===\n")
        final_state = None 

        try:
            for event in app.stream({"messages": full_messages}, thread):
                if isinstance(event, dict) and "state" in event:
                    final_state = event["state"]

                # Print the event type/name if available
                if hasattr(event, 'get') and callable(event.get):
                    print(f"\n--- Step: {event.get('name', 'Unknown')} ---")
                else:
                    print("\n--- New Event ---")
                
                # Print the event structure to understand what's available
                print(f"Event keys: {list(event.keys()) if hasattr(event, 'keys') else 'No keys'}")
                
                # Safely process the event
                if isinstance(event, dict):
                    for k, v in event.items():
                        if k == "state" or k == "name":
                            continue
                        
                        print(f"\n--- Processing: {k} ---")
                        
                        if isinstance(v, dict) and "messages" in v:
                            print("\nMessages:")
                            for msg in v["messages"]:
                                if isinstance(msg, HumanMessage):
                                    print(f"Human: {msg.content}")
                                elif isinstance(msg, AIMessage):
                                    print(f"AI: {msg.content}")
                                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                                        print(f"  Tool Calls: {len(msg.tool_calls)}")
                                        for tool_call in msg.tool_calls:
                                            print(f"    - {tool_call['name']}: {tool_call['args']}")
                                elif isinstance(msg, SystemMessage):
                                    print(f"System: {msg.content[:50]}...")
                                elif isinstance(msg, ToolMessage):
                                    print(f"Tool: {msg.content[:100]}...")
                                else:
                                    print(f"Other ({type(msg).__name__}): {msg}")
                        
                        if isinstance(v, dict):
                            if "retrieved_law_texts" in v and v["retrieved_law_texts"]:
                                print(f"\nRetrieved {len(v['retrieved_law_texts'])} law texts")
                                
                            if "final_answer" in v and v["final_answer"]:
                                print(f"\nFinal Answer available: {len(v['final_answer'])} chars")
                                
                            if "error_message" in v and v["error_message"]:
                                print(f"\nError: {v['error_message']}")
                else:
                    print(f"Event is not a dictionary: {type(event)}")
        except Exception as e:
            print(f"\n=== Error during execution: {e} ===\n")
        finally:
            if final_state and "messages" in final_state:
                conversation_history.append(new_message)
                ai_messages = [msg for msg in final_state["messages"] if isinstance(msg, AIMessage)]
                if ai_messages:
                    conversation_history.extend(ai_messages)
                
                # Optional: Keep history to a reasonable size (last 10 turns)
                if len(conversation_history) > 20:
                    conversation_history = conversation_history[-20:]
                
                # Display current conversation state
                # print("\n=== Current Conversation History ===")
                # for i, msg in enumerate(conversation_history[-6:]):  # Show last 3 turns
                #     if isinstance(msg, HumanMessage):
                #         print(f"Human: {msg.content}")
                #     elif isinstance(msg, AIMessage):
                #         print(f"AI: {msg.content[:150]}..." if len(msg.content) > 150 else f"AI: {msg.content}")
                # print("===============================")

            print("\n=== Graph Execution Completed ===\n")
            print(f"\n=== Token Usage Statistics ===")
            print(f"Input tokens: {input_tokens}")
            print(f"Output tokens: {output_tokens}")
            print(f"Total tokens: {input_tokens + output_tokens}")
            print("===============================")
