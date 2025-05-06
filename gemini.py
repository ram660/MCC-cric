import os
import json
import re # Make sure re is imported if extracting match_id from filename
from pathlib import Path
from dotenv import load_dotenv
import sqlite3 # Needed for DB error handling

# --- LangChain Core Imports ---
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate # Added PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda # Added RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage # Potentially for agent memory/state

# --- LLM Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
# --- Embedding Imports ---
from langchain_huggingface import HuggingFaceEmbeddings

# --- SQL Agent Imports ---
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent

# --- RAG Imports ---
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Agent & Tool Imports ---
from langchain.agents import Tool, AgentExecutor, create_react_agent
# For standard agent prompts, use a basic template instead of hub


# --- Configuration ---
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY needed. Set it in the .env file.")

JSON_WITH_SUMMARIES_DIR = "./output_json_with_summaries/" # Source for RAG data
VECTORSTORE_PERSIST_DIR = "./chroma_db_cricket_updated"        # Vectorstore location
DB_NAME = 'cricket_data_updated.db'                            # SQL Database location
MODEL_NAME = "gemini-2.0-flash"                 # LLM for agent and RAG

# --- Initialize LLM ---
# Shared LLM instance for SQL agent, RAG chain, and Router agent
llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    google_api_key=API_KEY,
    temperature=0, # Generally keep low for reliability, maybe slightly higher for RAG summary?
    # safety_settings=SAFETY_SETTINGS # Defined in previous script if needed
)
print("Gemini LLM initialized.")

# --- PART 1: Setup SQL Agent (from previous script, simplified) ---

print("Setting up SQL Database connection...")
if not os.path.exists(DB_NAME):
    raise FileNotFoundError(f"Database file '{DB_NAME}' not found. Run loading script first.")
db_uri = f"sqlite:///{DB_NAME}"
db = SQLDatabase.from_uri(db_uri)
print(f"Connected to SQL Database: {db_uri}")

SQL_AGENT_PREFIX = """
You are an agent designed to interact with a SQL database containing cricket match statistics.
Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 10 results.
You can order the results by a relevant column to return the most interesting examples in the database.
You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the schema description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

IMPORTANT: You MUST return ONLY the raw SQL query with NO formatting characters whatsoever. Do NOT use backticks, triple backticks, or any markdown formatting like ```sql around the query. The query should be plain text only.

The database includes tables for best performances:
- batting_performances: Contains the best batting performances with player_name, runs, balls, fours, sixes, and strike_rate
- bowling_performances: Contains the best bowling performances with player_name, overs, maidens, runs_conceded, wickets, and economy

For questions about best batsmen or best bowlers, use these tables to provide accurate information.

**Instructions for answering:**
- Based ONLY on the query results, answer the user's question.
- If the query returns no results or doesn't contain the necessary information, state clearly: "Based on the available structured match data, I cannot answer that question."
- Focus on factual data retrieval.
- For questions about best performances, use the batting_performances and bowling_performances tables.
- Examples of performance queries: "Who are the top 5 batsmen by runs?", "Who has the best bowling figures?", "Which batsman has the highest strike rate?", "Who took the most wickets in a match?"
"""

print("Creating SQL Agent...")
sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# Create the SQL agent with custom output handling
sql_agent_executor = create_sql_agent(
    llm=llm,
    toolkit=sql_toolkit,
    prefix=SQL_AGENT_PREFIX, # Specific instructions for the SQL agent
    verbose=False, # Keep SQL agent less verbose now, router will be verbose
    handle_parsing_errors=True
)
print("SQL Agent created.")

# --- PART 2: Setup RAG Pipeline (from previous snippet) ---

print("Setting up RAG pipeline...")
# 1. Load Summaries (assuming this part works as shown previously)
all_summaries_data = []
# ... (Paste the summary loading code from previous response here) ...
# --- Start of summary loading code ---
print("Loading summaries from JSON files...")
json_files_path = Path(JSON_WITH_SUMMARIES_DIR)
if not json_files_path.is_dir(): raise FileNotFoundError(f"Directory not found: {JSON_WITH_SUMMARIES_DIR}")
json_files = list(json_files_path.glob("*.json"))
if not json_files: raise FileNotFoundError(f"No JSON files found in {JSON_WITH_SUMMARIES_DIR}")
print(f"Found {len(json_files)} JSON files.")
for filepath in json_files:
    try:
        with open(filepath, 'r', encoding='utf-8') as f: data = json.load(f)
        summary = data.get("match_summary")
        match_id = None
        match = re.search(r'Scorecard_(\d+)\.json', filepath.name) # Extract ID from filename
        if match: match_id = int(match.group(1))
        if summary and isinstance(summary, str) and summary.strip():
            metadata = {"source": filepath.name}
            if match_id: metadata["match_id"] = match_id
            doc = Document(page_content=summary, metadata=metadata)
            all_summaries_data.append(doc)
    except Exception as e: print(f"Error processing file {filepath.name}: {e}. Skipping.")
if not all_summaries_data: raise ValueError("No summaries extracted for RAG.")
print(f"Loaded {len(all_summaries_data)} summaries.")
# --- End of summary loading code ---


# 2. Chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs_for_vectorstore = text_splitter.split_documents(all_summaries_data)
print(f"Split into {len(docs_for_vectorstore)} chunks.")

# 3. Embeddings
# Using GTE-base, one of the best open-source embedding models
embeddings = HuggingFaceEmbeddings(
    model_name="thenlper/gte-base",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# 4. Vector Store
print(f"Creating/loading vector store at: {VECTORSTORE_PERSIST_DIR}")
if os.path.exists(VECTORSTORE_PERSIST_DIR):
     print("Loading existing vector store...")
     vectorstore = Chroma(persist_directory=VECTORSTORE_PERSIST_DIR, embedding_function=embeddings)
else:
     print("Creating new vector store...")
     vectorstore = Chroma.from_documents(documents=docs_for_vectorstore, embedding=embeddings, persist_directory=VECTORSTORE_PERSIST_DIR)
print("Vector store ready.")

# 5. Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 chunks
print("Retriever created.")

# 6. RAG Chain Definition
RAG_PROMPT_TEMPLATE = """
You are the Maple Cricket Club League Assistant.
Answer the following question based ONLY on the provided context, which contains summaries of cricket matches.
If the context doesn't contain the answer, state clearly: "Based on the available match summaries, I cannot answer that question."

For questions about specific statistics, best batsmen, or best bowlers, you should suggest using the SQL database which contains detailed performance data including:
- Best batting performances with player name, runs, balls faced, fours, sixes, and strike rate
- Best bowling performances with player name, overs bowled, maidens, runs conceded, wickets taken, and economy rate

Context:
{context}

Question: {question}

Answer:"""
rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

def format_docs(docs):
    return "\n\n".join(f"Source: {doc.metadata.get('source', 'N/A')}\n{doc.page_content}" for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)
print("RAG Chain created.")


# --- PART 3: Define Tools for the Router Agent ---

print("Defining tools for router agent...")

# Custom function to handle SQL agent output and strip any backticks
def clean_sql_output(query):
    # Function to clean SQL output by removing backticks and code formatting
    try:
        # Call the SQL agent executor
        result = sql_agent_executor.invoke({"input": query})

        # Clean the output if it's in the expected format
        if isinstance(result, dict) and 'output' in result:
            # Remove any backticks or SQL formatting
            cleaned_output = re.sub(r'```sql|```|`', '', result['output']).strip()
            return cleaned_output
        return str(result)
    except Exception as e:
        return f"Error processing SQL query: {str(e)}"

sql_tool = Tool(
    name="Cricket_Stats_SQL_Database",
    func=clean_sql_output, # Use our custom wrapper function
    description="""Use this tool ONLY for questions requiring specific numerical statistics, scores, results, counts, averages, MAX/MIN values, or direct lookups from the cricket match database TABLES (matches, innings, batting_stats, bowling_stats, batting_performances, bowling_performances). Examples: 'Who scored the most runs?', 'What was the score in match X?', 'List bowlers with economy under 4', 'How many matches did Team Y win?', 'Who are the top batsmen?', 'Who has the best bowling figures?', 'Which player has the highest strike rate?'. Input should be the user's original question about structured data.""",
    # Coroutines can improve performance but require async setup
    # coroutine=...
)

rag_tool = Tool(
    name="Cricket_Match_Summaries_And_Context",
    func=rag_chain.invoke, # Invoke the RAG chain
    description="""Use this tool ONLY for questions asking for textual summaries of matches, descriptions of events, qualitative information ('tell me about...', 'describe...', 'what happened in the match between...'), or general context about specific games based on generated text summaries. Do NOT use for specific stats calculation. Input should be the user's original question.""",
)

tools = [sql_tool, rag_tool]
print(f"Tools defined: {[tool.name for tool in tools]}")


# --- PART 4: Create the Router Agent ---

# We'll use a standard ReAct agent prompt. This tells the LLM to Reason, Act (choose a tool), Observe the result.
# Using a basic ReAct prompt template directly
template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

IMPORTANT NOTES:
1. When using the Cricket_Stats_SQL_Database tool, make sure to pass ONLY the question text without any formatting or SQL code.
2. The SQL tool will handle creating and executing the appropriate SQL query.
3. If you see any errors about parsing SQL output, try again but be more explicit that the SQL tool should return ONLY the raw SQL query with no formatting.

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
prompt = PromptTemplate.from_template(template)


print("Creating router agent...")
# Create the agent that can use the tools
router_agent = create_react_agent(llm, tools, prompt)

# Create the AgentExecutor that runs the agent loop
router_agent_executor = AgentExecutor(
    agent=router_agent,
    tools=tools,
    verbose=True, # See the router's thought process
    handle_parsing_errors=True,
    max_iterations=7, # Increase max iterations to allow for retries
    early_stopping_method="force" # Force stop after max_iterations
)
print("Router agent executor created.")


# --- PART 5: Run the Chat Loop ---

print("\n--- Starting Hybrid Cricket Assistant (type 'exit' to quit) ---")
while True:
    query = input("Ask your cricket question: ")
    if query.lower() == 'exit':
        break
    if not query:
        continue

    try:
        # Use the ROUTER agent executor
        response = router_agent_executor.invoke({"input": query})
        print("\nAssistant:")
        # Output should be in 'output' key for AgentExecutor
        if isinstance(response, dict) and 'output' in response:
             # Clean any potential backticks from the final output as well
             cleaned_output = re.sub(r'```sql|```|`', '', response['output']).strip()
             print(cleaned_output)
        else:
             # Fallback if structure is unexpected
             print(response)
        print("-" * 30)

    except Exception as e:
        print(f"\nAn error occurred in the router agent: {e}")
        if isinstance(e, sqlite3.Error):
             print("A database error occurred. Ensure the DB file is not locked.")
        elif "parsing" in str(e).lower() or "backtick" in str(e).lower():
             print("There was an issue with SQL formatting. Try asking your question again.")
             print("For best results with performance questions, try phrases like:")
             print("- 'Show me the top batsmen by runs'")
             print("- 'Who are the best bowlers based on wickets taken?'")
        else:
             print("The router agent might have struggled choosing a tool or the chosen tool failed.")
             print("Check the verbose logs ('Thought:', 'Action:', 'Observation:') for details.")
        print("-" * 30)


print("\nChat finished.")