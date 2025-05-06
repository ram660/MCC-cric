import streamlit as st
import google.generativeai as genai
import os
import sqlite3
import pandas as pd
from dotenv import load_dotenv
import logging
from pathlib import Path

# --- Configuration ---
APP_DIR = Path(__file__).parent
DB_NAME = 'cricket_data_updated.db'
GENERATIVE_MODEL_NAME = "gemini-2.0-flash"
LOGO_URL = "https://www.svgrepo.com/show/38490/cricket.svg"
TEAM_LOGO_URL = "https://example.com/warriors_logo.png"  # Replace with your team logo URL

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Page Configuration and Styling ---
st.set_page_config(
    page_title="Warriors Cricket Assistant",
    page_icon="🏏",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #f0f2f6;
}
.stApp {
    max-width: 1200px;
    margin: 0 auto;
}
.st-emotion-cache-1y4p8pa {
    align-items: center;
}
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
}
.chat-message.user {
    background-color: #e6f3ff;
}
.chat-message.assistant {
    background-color: #f0f0f0;
}
.chat-message .avatar {
    width: 20%;
}
.chat-message .avatar img {
    max-width: 78px;
    max-height: 78px;
    border-radius: 50%;
    object-fit: cover;
}
.chat-message .message {
    width: 80%;
    padding: 0 1.5rem;
}
.performance-card {
    background-color: #ffffff;
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 10px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}
.performance-card h4 {
    color: #b21f24;
    margin-top: 0;
}
.team-header {
    display: flex;
    align-items: center;
}
.team-header img {
    margin-right: 15px;
}
</style>
""", unsafe_allow_html=True)

def configure_gemini():
    """Configure the Google Generative AI API."""
    try:
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("GOOGLE_API_KEY not found in environment variables. Gemini features will be disabled.")
            return False
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        logger.error(f"Error configuring Gemini: {e}")
        return False

def get_db_connection():
    """Get a connection to the SQLite database."""
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        return None

def get_best_batting_performances(limit=5):
    """Get the best batting performances from the database."""
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        cursor = conn.cursor()
        query = """
        SELECT bp.player_name, bp.runs, bp.balls, bp.fours, bp.sixes, bp.strike_rate, m.team1, m.team2, m.date
        FROM batting_performances bp
        JOIN matches m ON bp.match_id = m.match_id
        ORDER BY bp.runs DESC
        LIMIT ?
        """
        cursor.execute(query, (limit,))
        performances = cursor.fetchall()
        conn.close()
        return performances
    except Exception as e:
        logger.error(f"Error fetching batting performances: {e}")
        if conn:
            conn.close()
        return []

def get_best_bowling_performances(limit=5):
    """Get the best bowling performances from the database."""
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        cursor = conn.cursor()
        query = """
        SELECT bp.player_name, bp.overs, bp.maidens, bp.runs_conceded, bp.wickets, bp.economy, m.team1, m.team2, m.date
        FROM bowling_performances bp
        JOIN matches m ON bp.match_id = m.match_id
        ORDER BY bp.wickets DESC, bp.economy ASC
        LIMIT ?
        """
        cursor.execute(query, (limit,))
        performances = cursor.fetchall()
        conn.close()
        return performances
    except Exception as e:
        logger.error(f"Error fetching bowling performances: {e}")
        if conn:
            conn.close()
        return []

def search_performances(query, limit=10):
    """Search for performances based on player name."""
    conn = get_db_connection()
    if not conn:
        return [], []
    
    try:
        cursor = conn.cursor()
        # Search batting performances
        batting_query = """
        SELECT bp.player_name, bp.runs, bp.balls, bp.fours, bp.sixes, bp.strike_rate, m.team1, m.team2, m.date
        FROM batting_performances bp
        JOIN matches m ON bp.match_id = m.match_id
        WHERE bp.player_name LIKE ?
        ORDER BY bp.runs DESC
        LIMIT ?
        """
        cursor.execute(batting_query, (f"%{query}%", limit))
        batting_results = cursor.fetchall()
        
        # Search bowling performances
        bowling_query = """
        SELECT bp.player_name, bp.overs, bp.maidens, bp.runs_conceded, bp.wickets, bp.economy, m.team1, m.team2, m.date
        FROM bowling_performances bp
        JOIN matches m ON bp.match_id = m.match_id
        WHERE bp.player_name LIKE ?
        ORDER BY bp.wickets DESC, bp.economy ASC
        LIMIT ?
        """
        cursor.execute(bowling_query, (f"%{query}%", limit))
        bowling_results = cursor.fetchall()
        
        conn.close()
        return batting_results, bowling_results
    except Exception as e:
        logger.error(f"Error searching performances: {e}")
        if conn:
            conn.close()
        return [], []

def get_chat_response(prompt, chat_history):
    """Get a response from the Gemini model."""
    try:
        model = genai.GenerativeModel(GENERATIVE_MODEL_NAME)
        
        # Format chat history for Gemini
        formatted_history = []
        for message in chat_history:
            if message["role"] == "user":
                formatted_history.append({'role': 'user', 'parts': [message["content"]]})
            else:
                formatted_history.append({'role': 'model', 'parts': [message["content"]]})
        
        # If no history, start a new chat
        if not formatted_history:
            chat = model.start_chat()
        else:
            chat = model.start_chat(history=formatted_history)
        
        # Get response
        response = chat.send_message(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error getting chat response: {e}")
        return f"Sorry, I encountered an error: {str(e)}"

# --- Main App Logic ---

# Header
col1, col2 = st.columns([1, 6], gap="small")
with col1:
    st.image(LOGO_URL, width=80)
with col2:
    st.markdown("<h1 style='margin-bottom:0; margin-top: 10px; color:#b21f24;'>Warriors Cricket Club</h1>",
               unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top:-5px; color:#555; font-size:1.3rem;'>Cricket Assistant</h3>",
               unsafe_allow_html=True)
st.divider()

# Configure Gemini
gemini_configured = configure_gemini()

# Tabs for different sections
tab1, tab2, tab3 = st.tabs(["Chat", "Best Performances", "Player Search"])

with tab1:
    if gemini_configured:
        st.success("Cricket Assistant Ready! Ask me anything about the Warriors cricket team.")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
            # Add initial message
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Hi there! I'm the Warriors Cricket Assistant. How can I help you with our team data today? 🏏"
            })
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Handle user input
        if prompt := st.chat_input("Ask about matches, players, stats..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("Thinking...")
                
                # Get database information for context
                batting_performances = get_best_batting_performances(3)
                bowling_performances = get_best_bowling_performances(3)
                
                # Create context from database
                context = "Here's some information from our database:\n\n"
                context += "Top Batting Performances:\n"
                for bp in batting_performances:
                    context += f"- {bp['player_name']}: {bp['runs']} runs off {bp['balls']} balls with {bp['fours']} fours and {bp['sixes']} sixes (SR: {bp['strike_rate']})\n"
                
                context += "\nTop Bowling Performances:\n"
                for bp in bowling_performances:
                    context += f"- {bp['player_name']}: {bp['wickets']} wickets for {bp['runs_conceded']} runs in {bp['overs']} overs (Economy: {bp['economy']})\n"
                
                # Create full prompt
                full_prompt = f"""
                You are the Warriors Cricket Club Assistant. Answer the following question based on the provided context and your knowledge of cricket.
                
                Context:
                {context}
                
                Question: {prompt}
                
                Answer the question in a helpful and informative way. If you don't have enough information, say so.
                """
                
                # Get response
                response = get_chat_response(full_prompt, st.session_state.messages)
                message_placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.warning("Gemini API is not configured. Please set the `GOOGLE_API_KEY` environment variable to enable the chat assistant.")

with tab2:
    st.header("Best Performances")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Batting Performances")
        batting_performances = get_best_batting_performances(10)
        
        if batting_performances:
            for bp in batting_performances:
                with st.container(border=True):
                    st.markdown(f"### {bp['player_name']}")
                    st.markdown(f"**{bp['runs']} runs** off {bp['balls']} balls")
                    st.markdown(f"**Boundaries:** {bp['fours']} fours, {bp['sixes']} sixes")
                    st.markdown(f"**Strike Rate:** {bp['strike_rate']}")
                    st.markdown(f"**Match:** {bp['team1']} vs {bp['team2']}")
                    st.markdown(f"**Date:** {bp['date']}")
        else:
            st.info("No batting performances found.")
    
    with col2:
        st.subheader("Bowling Performances")
        bowling_performances = get_best_bowling_performances(10)
        
        if bowling_performances:
            for bp in bowling_performances:
                with st.container(border=True):
                    st.markdown(f"### {bp['player_name']}")
                    st.markdown(f"**{bp['wickets']} wickets** for {bp['runs_conceded']} runs")
                    st.markdown(f"**Overs:** {bp['overs']}")
                    st.markdown(f"**Economy:** {bp['economy']}")
                    st.markdown(f"**Maidens:** {bp['maidens']}")
                    st.markdown(f"**Match:** {bp['team1']} vs {bp['team2']}")
                    st.markdown(f"**Date:** {bp['date']}")
        else:
            st.info("No bowling performances found.")

with tab3:
    st.header("Player Search")
    
    search_query = st.text_input("Search for a player", "")
    
    if search_query:
        batting_results, bowling_results = search_performances(search_query)
        
        if not batting_results and not bowling_results:
            st.info(f"No performances found for '{search_query}'")
        else:
            if batting_results:
                st.subheader("Batting Performances")
                batting_df = pd.DataFrame(batting_results)
                st.dataframe(batting_df, use_container_width=True)
            
            if bowling_results:
                st.subheader("Bowling Performances")
                bowling_df = pd.DataFrame(bowling_results)
                st.dataframe(bowling_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #555; font-size: 0.9rem;'>"
    "Warriors Cricket Club Assistant | Powered by Google Gemini"
    "</div>",
    unsafe_allow_html=True
)
