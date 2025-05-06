# Maple Cricket Club Assistant

An AI-powered cricket analysis assistant that provides insights about Maple Cricket Club matches using Streamlit and Google's Gemini API.

## Project Structure
```
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── data/              # JSON match data files
│   └── *.json         # Individual match data files
├── pdf/               # PDF scorecards
│   └── *.pdf         # Individual match scorecards
└── static/           # Static assets (images, etc.)
```

## Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd cricket-analysis
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up API Keys:**
   * Create a `.env` file in the root directory
   * Add your Google Gemini API key:
     ```
     GOOGLE_API_KEY=your_api_key_here
     ```

## Available Data

The project includes:
* **Match Data (JSON)**: Located in `data/` directory
  * Contains detailed match statistics
  * Player performances
  * Team scores and results
  * League schedules and assignments
* **Scorecards (PDF)**: Located in `pdf/` directory
  * Original match scorecards in PDF format
  * Mapped to corresponding JSON data files

## Running the Application

1. **Start the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **Access the interface:**
   * Open your browser to `http://localhost:8501`
   * Ask questions about:
     * Match results and statistics
     * Player performances
     * Team comparisons
     * Upcoming schedules and assignments

## Features

* **Natural Language Queries**: Ask questions about matches, players, and statistics
* **Data Analysis**: 
  * Match result analysis
  * Player performance tracking
  * Team statistics across the club
  * Schedule information
* **Interactive Interface**: User-friendly Streamlit interface for easy interaction
* **PDF Integration**: Links JSON data with original PDF scorecards

## Dependencies

Key dependencies include:
* `streamlit`: Web interface
* `google-generativeai`: Gemini AI integration
* `python-dotenv`: Environment configuration
* `PyPDF2`: PDF processing
* `langchain`: AI/ML pipeline components
* Additional utilities for web scraping and data processing

## Notes

* The application uses the Gemini API which requires an active internet connection
* PDF files in the `pdf/` directory are linked to corresponding JSON files in `data/`
* JSON data files contain processed and structured match information