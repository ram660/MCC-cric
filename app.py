import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
import logging # Import logging
from pathlib import Path # Import Path for better path handling

# --- IMPORTANT: Ensure data_loader.py contains CricketDataLoaderV2 ---
# --- AND that it accepts/uses 'model_name' for Google embeddings ---
try:
    from data_loader import CricketDataLoaderV2 as CricketDataLoader
except ImportError:
    st.error("Error: Could not import CricketDataLoaderV2 from data_loader.py. "
             "Make sure the file exists and the class name is correct.")
    st.stop()
except Exception as e:
    st.error(f"Error importing or initializing from data_loader.py: {e}. "
             "Ensure it handles the 'model_name' parameter for Google embeddings.")
    logger.error(f"Error during data_loader import/init: {e}", exc_info=True)
    st.stop()


# --- Configuration ---
# Use Path for robust path construction
APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
PERSIST_DIR = APP_DIR / "cricket_faiss_index" # Define persistence directory

# --- Embedding Configuration ---
EMBEDDING_PROVIDER = 'google' # Set provider to Google
# Explicitly define the Google embedding model
GOOGLE_EMBEDDING_MODEL = "models/text-embedding-004"

# --- Generative Model Configuration ---
# Recommended free model: gemini-1.5-flash-latest
GENERATIVE_MODEL_NAME = "gemini-2.0-flash"

# --- Visuals ---
LOGO_URL="https://www.svgrepo.com/show/38490/cricket.svg"


# Setup logging for the app
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Page Configuration and Styling ---
st.set_page_config(
    page_title="Maple Cricket Club Assistant",
    page_icon="🏏",
    layout="wide"
)

# Custom CSS (minor adjustments for better alignment if needed)
st.markdown("""
<style>
/* ... [Your existing CSS remains unchanged] ... */
.main {
    background-color: #f0f2f6; /* Slightly lighter grey */
}
.stApp {
    max-width: 1200px;
    margin: 0 auto;
}
/* Adjust vertical alignment for header */
.st-emotion-cache-1y4p8pa {
    align-items: center; /* Vertically align items in the header columns */
}
h1, h2, h3 {
    color: #1a2a6c !important; /* Slightly deeper blue */
}
.stButton>button {
    background-color: #b21f24; /* Maple leaf red */
    color: #ffffff;
    border: none; /* Remove border */
    border-radius: 5px;
    padding: 0.5rem 1rem;
}
.stButton>button:hover {
    background-color: #8e191c; /* Darker red on hover */
    color: #ffffff;
}
.stTextInput>div>div>input {
    border: 1px solid #1a2a6c;
    border-radius: 5px;
}
.stChatMessage {
    background-color: #ffffff;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
/* Sidebar styling */
.stSidebar {
    background-color: #e9ecef;
}
.stSidebar h3 {
     color: #1a2a6c !important;
}
</style>
""", unsafe_allow_html=True)

# --- Initialization Functions ---

@st.cache_resource
def init_data_loader():
    """Initializes the CricketDataLoaderV2 instance, configured for the specified embedding provider."""
    logger.info(f"Initializing CricketDataLoaderV2 with data_dir='{DATA_DIR}', persist_dir='{PERSIST_DIR}', provider='{EMBEDDING_PROVIDER}'")
    try:
        # Ensure data directory exists before initializing loader
        if not DATA_DIR.is_dir():
             st.error(f"Data directory not found: {DATA_DIR}")
             logger.error(f"Data directory not found: {DATA_DIR}")
             return None # Return None to indicate failure

        # Create the persistence directory if it doesn't exist
        PERSIST_DIR.mkdir(parents=True, exist_ok=True)

        # --- Prepare arguments for CricketDataLoader ---
        loader_kwargs = {
            'data_dir': str(DATA_DIR),
            'persist_dir': str(PERSIST_DIR),
            'embedding_provider': EMBEDDING_PROVIDER,
            'verbose': True
            # Add other defaults if needed: 'chunk_size', 'chunk_overlap'
            # 'chunk_size': 1000,
            # 'chunk_overlap': 150,
        }

        # --- Pass the specific model name ONLY if using Google provider ---
        # --- Requires CricketDataLoaderV2 in data_loader.py to handle this ---
        if EMBEDDING_PROVIDER == 'google':
            loader_kwargs['model_name'] = GOOGLE_EMBEDDING_MODEL
            logger.info(f"Configuring CricketDataLoader for Google embeddings with model: {GOOGLE_EMBEDDING_MODEL}")
        # elif EMBEDDING_PROVIDER == 'huggingface':
             # Optionally specify a HuggingFace model here if needed
             # loader_kwargs['model_name'] = "hkunlp/instructor-large" # Example
             # logger.info(f"Configuring CricketDataLoader for HuggingFace embeddings.")
             # pass # Or let the loader use its internal default HF model

        # Initialize the loader with the prepared arguments
        loader = CricketDataLoader(**loader_kwargs)
        return loader

    except FileNotFoundError as e:
         st.error(f"Error initializing data loader: {e}")
         logger.error(f"Error initializing data loader: {e}", exc_info=True)
         return None
    except TypeError as e:
        st.error(f"Error initializing data loader: Potential mismatch in arguments passed to CricketDataLoaderV2. Does it accept 'model_name'? Error: {e}")
        logger.error(f"TypeError during data loader init, check arguments (especially 'model_name'): {e}", exc_info=True)
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during data loader initialization: {e}")
        logger.error(f"Unexpected error during data loader init: {e}", exc_info=True)
        return None

def configure_gemini():
    """Configure the Google Generative AI API."""
    try:
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            # Don't show error immediately, maybe user wants public data access only
            logger.warning("GOOGLE_API_KEY not found in environment variables. Gemini features (including Google Embeddings) will be disabled.")
            return False
        genai.configure(api_key=api_key)
        # Test if configuration is valid by listing models (optional, but good check)
        # try:
        #     models = [m for m in genai.list_models() if 'embedContent' in m.supported_generation_methods]
        #     logger.info(f"Available Google embedding models: {[m.name for m in models]}")
        # except Exception as list_e:
        #      logger.warning(f"Could not list Google models after configuration, API key might be invalid: {list_e}")
        #      st.warning("Could not verify Google API key by listing models. Ensure it's correct.")
        #      return False # Treat as not configured if listing fails

        logger.info("Google Generative AI API configured successfully.")
        return True
    except Exception as e:
        st.error(f"Error configuring Google Generative AI API: {e}")
        logger.error(f"Error configuring Google API: {e}", exc_info=True)
        return False

# --- Main App Logic ---

# Header
col1, col2 = st.columns([1, 6], gap="small") # Adjust column ratio if needed
with col1:
    st.image(LOGO_URL, width=80)
with col2:
    st.markdown("<h1 style='margin-bottom:0; margin-top: 10px; color:#b21f24;'>🍁 Maple Cricket Club</h1>",
               unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top:-5px; color:#555; font-size:1.3rem;'>League Assistant</h3>",
               unsafe_allow_html=True)
st.divider()

# Configure Gemini first, as it's needed for both embeddings and generation if provider is 'google'
gemini_configured = configure_gemini()

# Initialize data loader only if Gemini is configured (as Google Embeddings require it)
data_loader = None
data_loaded_successfully = False

if gemini_configured or EMBEDDING_PROVIDER != 'google': # Allow non-google provider even if API key fails
    data_loader = init_data_loader()
else:
    st.warning("Google API key not configured. Cannot initialize data loader with Google Embeddings.")
    logger.warning("Skipping data loader initialization because Google provider selected but API key is missing/invalid.")


# Load data if loader initialized successfully
if data_loader:
    # This call now handles loading from persistence or building the index
    with st.spinner("Loading and preparing cricket data... Please wait."):
        logger.info("Attempting to load/process data using data_loader.load_and_process_data()")
        vector_store = data_loader.load_and_process_data(force_rebuild=False) # Set force_rebuild=True to always rebuild

    if vector_store:
        data_loaded_successfully = True
        logger.info("Data loaded and vector store is ready.")
        st.toast("Cricket data ready!", icon="🏏")
    else:
        st.error("Failed to load or build the vector store. Check logs for details.")
        logger.error("data_loader.load_and_process_data() failed to return a valid vector store.")
else:
    # Error message handled within init_data_loader or the check above
    if EMBEDDING_PROVIDER == 'google' and not gemini_configured:
        st.error("Data loader requires Google Embeddings, but the Google API is not configured. Please set `GOOGLE_API_KEY`.")
    elif not data_loader and (gemini_configured or EMBEDDING_PROVIDER != 'google'):
         st.warning("Data loader could not be initialized. Assistant may not function correctly.")


# --- Sidebar Status Display ---
with st.sidebar:
    st.subheader("📊 Data & System Status")
    if data_loader:
        # Use the new get_status method
        status = data_loader.get_status()
        vs_status = status.get("vector_store_status", {})
        load_status = status.get("loading_status", {})

        st.markdown("**Data Source:**")
        st.info(f"`{os.path.basename(status.get('data_directory', 'N/A'))}`")

        st.markdown("**Vector Store:**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Status", "Ready ✅" if vs_status.get("initialized") else "Not Ready ❌")
        with col2:
            st.metric("Vectors", f"{vs_status.get('vector_count', 0):,}")

        st.markdown("**Embedding:**")
        # Display the specific model if available in status, otherwise show provider
        embed_provider = status.get('embedding_provider', 'N/A').capitalize()
        embed_model = status.get('embedding_model', 'Default') # Get specific model if returned by loader
        st.info(f"{embed_provider} ({embed_model})")


        with st.expander("File Loading Details"):
             st.write(f"- JSON Files Found: {load_status.get('total_json_files_in_dir', 'N/A')}")
             st.write(f"- Files Processed: {load_status.get('files_processed', 'N/A')}")
             st.write(f"- Files Loaded Ok: {load_status.get('files_successfully_loaded', 'N/A')}")
             st.write(f"- Files Failed/Skipped: {load_status.get('files_failed_or_skipped', 'N/A')}")
             st.write(f"- Items Processed: {load_status.get('total_items_processed', 'N/A')}")
             st.write(f"- Items Failed: {load_status.get('total_items_failed', 'N/A')}")

    else:
        st.warning("Data Loader not available.")

    st.divider()
    st.subheader("🤖 AI Model Status")
    if gemini_configured:
        st.success(f"Gemini Ready ({GENERATIVE_MODEL_NAME})")
    else:
        st.warning("Gemini API not configured. Check GOOGLE_API_KEY.")


# --- Chat Interface ---
if data_loaded_successfully and gemini_configured:
    st.success("Assistant Ready! Ask me anything about the Maple Cricket Club data.")

    # Initialize chat history and model
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "gemini_chat" not in st.session_state:
        try:
            logger.info(f"Initializing Gemini model: {GENERATIVE_MODEL_NAME}")
            model = genai.GenerativeModel(GENERATIVE_MODEL_NAME)

            # System Prompt / Initial Context for the LLM
            # (More concise, focuses on role and data source)
            initial_prompt = f"""
            You are the Maple Cricket Club League Assistant.
            Your primary role is to answer questions based *strictly* on the provided context about cricket matches, schedules, players, and statistics related to the club.
            The context comes from a specialized database indexed using Google's text-embedding-004 model.

            **Instructions:**
            1.  **Base answers ONLY on the provided context.** Do not use external knowledge or make assumptions.
            2.  If the context contains the answer, provide it clearly and concisely. Quote relevant details like scores, dates, player names accurately.
            3.  **If the context does *not* contain the information to answer the question, explicitly state that.** Say something like, "Based on the provided information, I cannot answer that question." or "The available data doesn't include details about [topic]."
            4.  Maintain a helpful and informative tone, focused on cricket data.
            5.  Do not hallucinate or invent information. Accuracy based on the context is paramount.
            """

            logger.info("Starting Gemini chat session.")
            # Start chat with the system prompt
            st.session_state.gemini_chat = model.start_chat(history=[
                {'role': 'user', 'parts': [initial_prompt]},
                {'role': 'model', 'parts': ["Understood. I am the Maple Cricket Club Assistant. I will answer questions strictly based on the provided context (indexed with Google text-embedding-004) and state when information is unavailable in the data."]}
            ])

            # Add the initial assistant message only if chat history is empty
            if not st.session_state.messages:
                 st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Hi there! I'm the Maple Cricket Club Assistant. How can I help you with our league data today? 🏏"
                })
                 logger.info("Added initial welcome message to chat.")

        except Exception as e:
            st.error(f"Failed to initialize Gemini chat model: {e}")
            logger.error(f"Failed to initialize Gemini chat model: {e}", exc_info=True)
            st.stop() # Stop execution if chat model fails

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Ask about matches, schedules, stats..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        logger.info(f"User query: {prompt}")

        # 1. Get relevant context using RAG
        context = "No relevant context found." # Default
        with st.spinner("Searching cricket database..."):
            try:
                # Use the get_relevant_context from the loader instance
                # Ensure the data_loader instance is valid before calling methods
                if data_loader:
                    context = data_loader.get_relevant_context(prompt, k=5) # Fetch top 5 chunks
                    logger.info(f"Retrieved context for query '{prompt}'. Length: {len(context)}")
                    # Optional: Show context for debugging
                    # with st.sidebar.expander("Retrieved Context"):
                    #      st.text(context if context else "None")
                else:
                    logger.error("Cannot retrieve context: data_loader is not initialized.")
                    st.error("Error: Data loader not available for context retrieval.")
                    context = "Error: Data loader not available."

            except Exception as e:
                 logger.error(f"Error retrieving context: {e}", exc_info=True)
                 st.error("Error searching the database.")
                 context = "Error retrieving context from the database."


        # 2. Construct full prompt for LLM
        #    (Include clear separation and instruction)
        full_prompt = f"""
        **User Question:**
        {prompt}

        ---
        **Relevant Context from Cricket Database (using text-embedding-004):**
        {context}
        ---

        **Instruction:** Based *only* on the context provided above, answer the user's question. If the context does not contain the answer, state that clearly. Do not add information not present in the context.
        """

        # 3. Get assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            try:
                logger.info("Sending prompt to Gemini.")
                # Ensure the chat session exists
                if "gemini_chat" in st.session_state:
                     # Send the combined prompt to the existing chat session
                    response = st.session_state.gemini_chat.send_message(full_prompt, stream=True)

                    full_response = ""
                    for chunk in response:
                        # Error handling for potential empty chunks or access issues
                        try:
                            if chunk.text:
                                full_response += chunk.text
                                message_placeholder.markdown(full_response + "▌") # Typing indicator
                        except ValueError:
                            # Sometime chunk generation gets interrupted, ignore value error.
                            logger.warning("ValueError accessing chunk text, skipping.")
                            pass
                        except Exception as e:
                            logger.error(f"Error processing chunk: {e}", exc_info=True)
                            pass # Avoid breaking the stream for minor chunk errors

                    message_placeholder.markdown(full_response) # Final response
                    logger.info(f"Gemini response received. Length: {len(full_response)}")
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    logger.error("Cannot send message: Gemini chat session not initialized.")
                    message_placeholder.error("Error: Chat session is not available.")
                    st.session_state.messages.append({"role": "assistant", "content": "Error: Chat session lost."})


            except genai.types.generation_types.BlockedPromptException as e:
                error_msg = "⚠️ My safety filters prevented processing that request. Please rephrase your question."
                logger.warning(f"BlockedPromptException: {e}")
                message_placeholder.warning(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            except Exception as e:
                error_msg = f"😕 Apologies, an error occurred while generating the response: {e}"
                logger.error(f"Error during Gemini generation: {e}", exc_info=True)
                message_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Fallback messages if prerequisites fail
elif not data_loaded_successfully:
     if data_loader: # Loader initialized but failed loading data
         st.warning("Cricket data could not be fully loaded or processed. The assistant is unavailable.")
     # If data_loader is None, the specific error (e.g. API key missing) was shown earlier.
elif not gemini_configured:
    st.warning("Gemini API is not configured. Please set the `GOOGLE_API_KEY` environment variable to enable the chat assistant.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #555; font-size: 0.9rem;'>"
    "Maple Cricket Club Assistant | Powered by Google Gemini & LangChain | Embeddings: text-embedding-004" # Updated footer
    "</div>",
    unsafe_allow_html=True
)