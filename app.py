import streamlit as st
import google.generativeai as genai
import json
import os
from dotenv import load_dotenv

# --- Configuration ---
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")  # Absolute path to data directory
MODEL_NAME = "gemini-2.0-flash" # Or "gemini-1.0-pro", "gemini-1.5-pro-latest" etc.
# Use a URL for the logo instead of a local file to avoid image loading issues
LOGO_URL = "https://raw.githubusercontent.com/ram660/warriors/main/static/warriors_logo.png"

# --- Helper Functions ---

def load_data(directory):
    """Loads and combines cricket data from all JSON files in the given directory."""
    all_data = []
    file_count = 0
    try:
        # Check if directory exists
        if not os.path.exists(directory):
            st.error(f"Error: Data directory '{directory}' not found. Make sure it exists.")
            return None

        # Get list of all JSON files
        json_files = [f for f in os.listdir(directory) if f.endswith('.json')]

        if not json_files:
            st.warning(f"No JSON files found in '{directory}'. Please add data files.")
            return None

        # Process each JSON file
        for filename in json_files:
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # If data is already a list, extend all_data with its contents
                    # Otherwise, append the entire data object
                    if isinstance(data, list):
                        all_data.extend(data)
                        file_count += 1
                    else:
                        all_data.append(data)
                        file_count += 1
            except json.JSONDecodeError as e:
                # Silently skip files with JSON errors
                continue
            except Exception as e:
                # Silently skip files with other errors
                continue

        if not all_data:
            st.error("No valid data could be loaded from any JSON files.")
            return None

        # Combine all data into a single JSON structure
        data_string = json.dumps(all_data, indent=2)
        return data_string
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

# Custom CSS for better UI with Warriors colors (gold and black)
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
/* Warriors colors */
h1, h2, h3 {
    color: #000000 !important;
}
.stButton>button {
    background-color: #ffc107;
    color: #000000;
}
.stTextInput>div>div>input {
    border-color: #ffc107;
}
</style>
""", unsafe_allow_html=True)

# Header with Warriors branding
col1, col2 = st.columns([1, 5])
with col1:
    try:
        # Try to use the Warriors logo from URL
        st.image("https://img.icons8.com/color/96/000000/cricket.png", width=80)
    except Exception as e:
        # Fallback to a generic cricket icon if there's any issue
        st.image("https://img.icons8.com/color/96/000000/cricket.png", width=80)
with col2:
    # Custom styled header with Warriors colors
    st.markdown("""<h1 style='margin-bottom:0; color:#000000;'>🏏 <span style='color:#ffc107;'>WHONNOCK</span> <span style='color:#000000;'>WARRIORS</span></h1>""", unsafe_allow_html=True)
    st.markdown("""<h3 style='margin-top:0; color:#666666; font-size:1.2rem;'>Cricket Team Assistant</h3>""", unsafe_allow_html=True)
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
            Your knowledge is STRICTLY LIMITED to the following JSON data containing match information, schedules, statistics, and scorecard information for the upcoming 2025 season.
            The schedule and assignments data is for the UPCOMING season, while match results are from past seasons.
            Do NOT use any external knowledge or real-time information. Answer questions based ONLY on the provided JSON data.

            Always refer to the team as "Warriors" or "Whonnock Warriors" and speak with enthusiasm about the team's performance.
            When discussing players, emphasize their contributions to the Warriors team.
            Use a friendly, energetic tone that reflects team spirit.

            **Data Types in the JSON:**
            The data contains several types of information for all teams in the league, including the Warriors:
            1. **Match Results & Scorecards**: Detailed information about completed matches from past seasons, including scores, player performances, and statistics
            2. **League Schedule**: Complete schedule of all matches for the UPCOMING 2025 season, including dates, times, and locations
            3. **League Assignments**: Ground setup, teardown, umpiring, and scoring duties for all teams for the UPCOMING 2025 season
            4. **League Statistics**: Overall performance metrics for all teams in the league

            **Instructions for Answering:**

            1.  **Direct Information:** If a question asks for information directly present (e.g., "What was the score in the match against Maple Mavericks?", "When is our next game?", "What's our schedule?"), retrieve and state that information with enthusiasm for Warriors' performance.

            2.  **Match Information:** For questions about match results, look for data with match details including:
                * Match results (wins/losses)
                * Player performances
                * Scores and statistics
                * Best performances

            3.  **Schedule Information:** For questions about schedules, look for data with "data_type": "league_schedule" which contains the UPCOMING 2025 season schedule:
                * Dates of all matches in the league for the upcoming season
                * Match times (Friday evening or Saturday slots)
                * Team matchups (e.g., "Mavericks vs Warriors")
                * Ground setup and teardown assignments for each match
                * Umpiring and scoring duties for each match
                * When answering about Warriors schedule, filter for matches where "Warriors" appears in the matchup
                * Always clarify that you're providing information about the UPCOMING 2025 season

            4.  **League Assignments:** For questions about team duties, look for data with "data_type": "league_assignments" which contains information for the UPCOMING 2025 season:
                * Ground setup assignments for all teams
                * Ground teardown assignments for all teams
                * Umpiring and scoring assignments for all teams
                * Games per week statistics for all teams
                * When answering about Warriors assignments, focus on the "Whonnock Warriors" column or entries
                * Always clarify that you're providing information about the UPCOMING 2025 season

            5.  **Aggregation/Summarization:** If a question requires summarizing or aggregating data across matches or players (e.g., "Who hit the most sixes for Warriors?", "What was Abhilash Reddy Gade's total score across all matches?", "How many wickets did Sudheer take in total?"), you MUST follow these steps:
                *   Identify the key entity (player name, statistic like 'sixes', 'wickets', 'runs').
                *   Carefully scan the batting/bowling cards and other statistics for *all relevant matches* in the JSON data.
                *   For 'sixes': Check the batting cards for the '6s' column.
                *   For 'wickets': Check the bowling cards for the 'W' column.
                *   For 'runs': Check the batting cards for the 'R' column.
                *   Aggregate the counts/stats for each relevant player across *all* the matches found in the data.
                *   Provide the final aggregated result with enthusiasm for Warriors' achievements (e.g., "Sharath Reddy Gade was outstanding for the Warriors, hitting the most sixes with a total of X across the provided matches!").

            6.  **Limitations:** If the specific information is not present *within the provided JSON*, clearly state that "Based on the available Warriors match data, I cannot definitively determine..." Do not guess or make assumptions.

            7.  **Warriors Focus:** You are primarily a Warriors team assistant, so focus on Warriors information by default. However, you can provide information about other teams when specifically asked. Always highlight Warriors team performances and achievements in your responses. Be positive and supportive of the Warriors even when discussing losses.

            8.  **Comparative Analysis:** When asked to compare Warriors with other teams, use the league-wide data to provide accurate comparisons of schedules, assignments, or statistics. Always frame your response to emphasize the Warriors' strengths while being factual.

            Here is the cricket data you have access to:
            ```json
            {cricket_data_str}
            ```

            I'm ready to answer your questions about the Whonnock Warriors cricket team! How can I help you today?
            """
            # Start the chat with the initial context. Gemini models usually handle history well.
            try:
                st.session_state.gemini_chat = model.start_chat(history=[
                     {'role':'user', 'parts': [initial_prompt]},
                     {'role':'model', 'parts': ["I'm ready to answer your questions about the Whonnock Warriors cricket team! How can I help you today?"]}
                ])
                # Add the initial assistant message to the display history
                st.session_state.messages.append({"role": "assistant", "content": "I'm ready to answer your questions about the Whonnock Warriors cricket team! How can I help you today?"})
            except Exception as e:
                # If the data is too large, try with a truncated version
                data_size = len(cricket_data_str)
                if data_size > 100000:  # If data is over ~100KB
                    truncated_data = json.dumps(json.loads(cricket_data_str)[:10], indent=2)  # Take only first 10 items
                    truncated_prompt = initial_prompt.replace(cricket_data_str, truncated_data)
                    st.session_state.gemini_chat = model.start_chat(history=[
                         {'role':'user', 'parts': [truncated_prompt]},
                         {'role':'model', 'parts': ["I'm ready to answer your questions about the Whonnock Warriors cricket team! How can I help you today?"]}
                    ])
                    st.session_state.messages.append({"role": "assistant", "content": "I'm ready to answer your questions about the Whonnock Warriors cricket team! How can I help you today?"})
                else:
                    raise e

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
st.markdown("<div style='text-align: center;'><span style='color: #ffc107; font-weight: bold;'>WHONNOCK WARRIORS</span> <span style='color: #666;'>Cricket Team | Powered by Gemini AI</span></div>", unsafe_allow_html=True)


