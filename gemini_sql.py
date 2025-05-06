import os
from dotenv import load_dotenv
import sqlite3

# --- Updated LangChain Imports ---
#from langchain_core.agents import AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent

# --- 1. Define the Agent Persona and Instructions ---
# Removed the sentence about embeddings as it's not relevant for the SQL agent
AGENT_PREFIX = """
You are the Maple Cricket Club League Assistant.
Your primary role is to answer questions based *strictly* on the provided context about cricket matches, schedules, players, and statistics related to the club.
The context comes from a specialized SQL database containing match scorecards.

**Instructions:**
1.  Base answers ONLY on the provided SQL database context. Do not use external knowledge or make assumptions.
2.  If the database contains the answer (based on the SQL query results), provide it clearly and concisely. Quote relevant details like scores, dates, player names accurately.
3.  If the database context does *not* contain the information to answer the question (e.g., the SQL query returns no results or doesn't support the question), explicitly state that. Say something like, "Based on the available match data, I cannot answer that question." or "The database doesn't include details about [topic]."
4.  Maintain a helpful and informative tone, focused on cricket data.
5.  Do not hallucinate or invent information. Accuracy based on the database context is paramount.
6.  **IMPORTANT:** When providing SQL queries to be executed (e.g., for the 'sql_db_query' action), provide **only the raw SQL code**, without any surrounding characters like backticks or Markdown code fences (```sql ... ```). Just the plain SQL statement.
"""

# --- 2. Load API Key ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Google API key not found. Please set GOOGLE_API_KEY in the .env file.")

# --- 3. Set up Database Connection ---
DB_NAME = 'cricket_data.db' # <<< Make sure this matches the DB file created
if not os.path.exists(DB_NAME):
    raise FileNotFoundError(f"Database file '{DB_NAME}' not found. Please run the loading script first.")

db_uri = f"sqlite:///{DB_NAME}"
db = SQLDatabase.from_uri(db_uri)
print(f"Connected to SQL Database: {db_uri}")
# print("Detected Tables:", db.get_table_names()) # Uncomment to verify tables

# --- 4. Initialize Gemini LLM ---
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0, # Keep low for factual tasks
    )
    print("Gemini 1.5 Flash LLM initialized successfully.")

    # --- 5. Create the SQL Agent with Persona ---
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        prefix=AGENT_PREFIX, # <<< Inject the persona and instructions here
        verbose=True,
        handle_parsing_errors=True
    )
    print("SQL Agent created successfully with custom instructions.")

    # --- 6. Ask Questions! ---
    print("\n--- Starting Maple Cricket Club League Assistant (type 'exit' to quit) ---")
    while True:
        query = input("Ask your cricket question: ")
        if query.lower() == 'exit':
            break
        if not query:
            continue

        try:
            response = agent_executor.invoke({"input": query})
            print("\nAssistant:") # Changed label to match persona
            if isinstance(response, dict) and 'output' in response:
                 print(response['output'])
            else:
                 print(response)
            print("-" * 30)

        except Exception as e:
            print(f"\nAn error occurred: {e}")
            if isinstance(e, sqlite3.Error):
                 print("A database error occurred. Ensure the DB file is not locked.")
            print("The agent might have struggled with generating or executing the SQL query.")
            print("Check the verbose logs ('Thought:', 'Action:', 'Observation:') for details.")
            print("-" * 30)


except Exception as e:
    print(f"Failed to initialize LLM or Agent: {e}")

print("\nChat finished.")