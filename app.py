import streamlit as st
import google.generativeai as genai
import json
import os
from dotenv import load_dotenv

# --- Configuration ---
DATA_DIR = ".\data"  # Updated to directory
MODEL_NAME = "gemini-2.0-flash" # Or "gemini-1.0-pro", "gemini-1.5-pro-latest" etc.

# --- Helper Functions ---

def load_data(directory):
    """Loads and combines cricket data from all JSON files in the given directory."""
    all_data = []
    try:
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                filepath = os.path.join(directory, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_data.append(data)
        # Combine all data into a single JSON structure (list of matches/files)
        data_string = json.dumps(all_data, indent=2)
        return data_string
    except FileNotFoundError:
        st.error(f"Error: Data directory '{directory}' not found. Make sure it exists.")
        return None
    except json.JSONDecodeError as e:
        st.error(f"Error: Could not decode JSON in one of the files. {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading data: {e}")
        return None

def configure_gemini():
    """Configures the Gemini API."""
    try:
        # Load environment variables from .env file
        load_dotenv()

        # Get API key from environment variable
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("Error: GOOGLE_API_KEY not found in environment variables. Please check your .env file.")
            return False

        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Error configuring Gemini API: {e}")
        return False

# --- Main Application Logic ---

st.set_page_config(page_title="Warriors Cricket Assistant", layout="wide")

# Custom CSS for better UI
st.markdown("""
<style>
.main {
    background-color: #f5f5f5;
}
.stApp {
    max-width: 1200px;
    margin: 0 auto;
}
.css-18e3th9 {
    padding-top: 2rem;
}
.stTitle {
    font-weight: bold;
    color: #1e3a8a;
}
</style>
""", unsafe_allow_html=True)

# Header with Warriors branding
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://img.icons8.com/color/96/000000/cricket.png", width=80)
with col2:
    st.title("🏏 Whonnock Warriors Cricket Assistant")
    st.caption("Your personal assistant for Warriors team stats, player performance, and match insights.")

# Load Cricket Data
cricket_data_str = load_data(DATA_DIR)

# Configure Gemini
gemini_configured = configure_gemini()

if cricket_data_str and gemini_configured:
    # Initialize chat history in session state if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "gemini_chat" not in st.session_state:
        try:
            model = genai.GenerativeModel(MODEL_NAME)
            # System Instruction / Initial Context for the Chat
            # IMPORTANT: Tell the model its role and limitations clearly.
            initial_prompt = f"""
            You are a highly specialized cricket assistant for the Whonnock Warriors cricket team.
            Your knowledge is STRICTLY LIMITED to the following JSON data containing commentary and scorecard information for several cricket matches involving the Warriors team.
            Do NOT use any external knowledge or real-time information. Answer questions based ONLY on the provided JSON data.

            Always refer to the team as "Warriors" or "Whonnock Warriors" and speak with enthusiasm about the team's performance.
            When discussing players, emphasize their contributions to the Warriors team.

            **Instructions for Answering:**

            1.  **Direct Information:** If a question asks for information directly present (e.g., "What was the score in the match against Maple Mavericks?"), retrieve and state that information with enthusiasm for Warriors' performance.
            2.  **Aggregation/Summarization (Crucial):** If a question requires summarizing or aggregating data across matches or players (e.g., "Who hit the most sixes for Warriors?", "What was Abhilash Reddy Gade's total score across all matches?", "How many wickets did Sudheer take in total?"), you MUST follow these steps:
                *   Identify the key entity (player name, statistic like 'sixes', 'wickets', 'runs').
                *   Carefully scan the 'commentary' list and batting/bowling cards for *all relevant matches* in the JSON data.
                *   For 'sixes': Look for commentary entries containing the word "SIX" or check the batting cards for the '6s' column. Attribute the six to the batsman mentioned.
                *   For 'wickets': Look for commentary entries indicating an "OUT" status or check the bowling cards for the 'W' column. Attribute the wicket to the bowler mentioned.
                *   For 'runs': Check the batting cards for the 'R' column or extract from commentary when necessary.
                *   Aggregate the counts/stats for each relevant player across *all* the matches found in the data.
                *   Provide the final aggregated result with enthusiasm for Warriors' achievements (e.g., "Sharath Reddy Gade was outstanding for the Warriors, hitting the most sixes with a total of X across the provided matches!").
            3.  **Limitations:** If the specific information is not present *within the provided JSON*, clearly state that "Based on the available Warriors match data, I cannot definitively determine..." Do not guess or make assumptions.
            4.  **Scope:** Remember, your analysis is confined ONLY to the JSON content below.
            5.  **Team Focus:** Always highlight Warriors team performances and achievements in your responses.

            Here is the cricket data you have access to:
            ```json
            {cricket_data_str}
            ```

            I'm ready to answer your questions about the Whonnock Warriors cricket team! How can I help you today?
            """
            # Start the chat with the initial context. Gemini models usually handle history well.
            st.session_state.gemini_chat = model.start_chat(history=[
                 {'role':'user', 'parts': [initial_prompt]},
                 {'role':'model', 'parts': ["I'm ready to answer your questions about the Whonnock Warriors cricket team! How can I help you today?"]}
            ])
            # Add the initial assistant message to the display history
            st.session_state.messages.append({"role": "assistant", "content": "I'm ready to answer your questions about the Whonnock Warriors cricket team! How can I help you today?"})

        except Exception as e:
            st.error(f"Failed to initialize Gemini chat model: {e}")
            st.stop() # Stop execution if model fails to load


    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask about Warriors matches, player stats, or team performance..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            try:
                # Send the user's prompt to the ongoing chat
                response = st.session_state.gemini_chat.send_message(prompt, stream=True)
                full_response = ""
                for chunk in response:
                     # Handle potential errors in chunk processing if necessary
                     try:
                         full_response += chunk.text
                         message_placeholder.markdown(full_response + "▌") # Add cursor effect
                     except Exception as e:
                         pass # Continue with the next chunk

                message_placeholder.markdown(full_response) # Display final response

            except genai.types.generation_types.BlockedPromptException as bpe:
                 full_response = "My apologies, I cannot respond to that request due to safety filters. Please try a different question."
                 message_placeholder.error(full_response)
            except Exception as e:
                full_response = f"An error occurred: {e}"
                message_placeholder.error(full_response)
                st.exception(e) # Show detailed error in Streamlit console/app

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})


elif not cricket_data_str:
    st.warning("Cricket data could not be loaded. Please check the directory path and file formats.")
elif not gemini_configured:
    st.warning("Gemini API could not be configured. Please check your API key in secrets.")

# Add a footer with team information
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>Whonnock Warriors Cricket Team | Powered by Gemini AI</div>", unsafe_allow_html=True)
