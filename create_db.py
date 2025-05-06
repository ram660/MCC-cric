import json
import pandas as pd # Still useful for initial parsing/cleaning
import os
import glob
import re
import sqlite3

# --- Configuration ---
JSON_DIR = r'C:\Users\ramma\Downloads\New folder\data' # <<< Directory containing your JSON files
DB_NAME = 'cricket_data_updated.db'   # <<< Name for the SQLite database file - UPDATED to match gemini.py

# --- Helper Function to Extract Match ID ---
def extract_match_id(filename):
    match = re.search(r'Scorecard_(\d+)\.json', filename)
    if match:
        return int(match.group(1))
    return None

# --- Helper Function to Parse Score String ---
def parse_score(score_str):
    """Parses score strings like '116', '211/8' into (runs, wickets)."""
    score_str = str(score_str).strip()
    total_runs = 0
    total_wickets = 0 # Default to 0 if not specified
    if '/' in score_str:
        parts = score_str.split('/')
        try:
            total_runs = int(parts[0])
            total_wickets = int(parts[1])
        except (ValueError, IndexError):
            print(f"Warning: Could not parse score/wickets from '{score_str}'")
    else:
        try:
            total_runs = int(score_str)
            # If only runs are given, we don't know wickets for sure, leave as 0
            # Or could assume 10 if it was an all-out situation, but 0 is safer.
        except ValueError:
            print(f"Warning: Could not parse runs from '{score_str}'")
    return total_runs, total_wickets

# --- Helper Function to Parse Overs String ---
def parse_overs_float(over_str):
     """Cleans 'overs' field (e.g., '0.4 ov', '5 ov', '19\nov') -> float"""
     try:
        # Handle potential multiline strings and remove "ov" suffix
        cleaned_str = str(over_str).lower().replace(' ov', '').strip()
        return float(cleaned_str)
     except (ValueError, TypeError):
        print(f"Warning: Could not parse overs '{over_str}' to float. Using 0.0.")
        return 0.0

# --- Database Setup ---
print(f"Connecting to database: {DB_NAME}")
# Connect to SQLite database (creates the file if it doesn't exist)
conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()

# Enable foreign key support (good practice)
cursor.execute("PRAGMA foreign_keys = ON;")

# --- Create Tables (if they don't exist) ---
# Use INTEGER PRIMARY KEY for auto-incrementing IDs where needed
cursor.execute('''
CREATE TABLE IF NOT EXISTS matches (
    match_id INTEGER PRIMARY KEY,
    filename TEXT,
    team1 TEXT,
    team2 TEXT,
    toss_winner TEXT,
    toss_decision TEXT,
    venue TEXT,
    date TEXT,
    result TEXT
)''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS innings (
    inning_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL,
    inning_number INTEGER NOT NULL,
    batting_team TEXT,
    bowling_team TEXT,
    total_runs INTEGER,
    total_wickets_lost INTEGER,
    overs_played REAL,
    raw_total_score_string TEXT,
    FOREIGN KEY (match_id) REFERENCES matches (match_id)
)''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS batting_stats (
    batting_stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
    inning_id INTEGER NOT NULL,
    match_id INTEGER NOT NULL, -- For easier joins directly to matches
    player_name TEXT,
    status TEXT,
    runs INTEGER,
    balls INTEGER,
    fours INTEGER,
    sixes INTEGER,
    strike_rate REAL,
    FOREIGN KEY (inning_id) REFERENCES innings (inning_id),
    FOREIGN KEY (match_id) REFERENCES matches (match_id)
)''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS bowling_stats (
    bowling_stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
    inning_id INTEGER NOT NULL,
    match_id INTEGER NOT NULL, -- For easier joins directly to matches
    player_name TEXT,
    overs REAL,
    maidens INTEGER,
    runs_conceded INTEGER,
    wickets INTEGER,
    economy REAL,
    FOREIGN KEY (inning_id) REFERENCES innings (inning_id),
    FOREIGN KEY (match_id) REFERENCES matches (match_id)
)''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS fall_of_wickets (
    fow_id INTEGER PRIMARY KEY AUTOINCREMENT,
    inning_id INTEGER NOT NULL,
    match_id INTEGER NOT NULL, -- For easier joins directly to matches
    score_at_fall INTEGER,
    wicket_number INTEGER,
    player_out TEXT,
    over_at_fall REAL,
    FOREIGN KEY (inning_id) REFERENCES innings (inning_id),
    FOREIGN KEY (match_id) REFERENCES matches (match_id)
)''')

print("Database tables ensured.")

# --- Find and Process JSON Files ---
json_files = glob.glob(os.path.join(JSON_DIR, 'Scorecard_*.json'))

if not json_files:
    print(f"Error: No 'Scorecard_*.json' files found in directory: {JSON_DIR}")
    # Create a sample match if no files found
    print("Creating a sample match record...")
    
    # Insert a sample match
    sample_match_id = 12345
    cursor.execute('''
        INSERT INTO matches (match_id, filename, team1, team2, toss_winner, toss_decision, venue, date, result)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        sample_match_id,
        "sample_match.json",
        "Warriors",
        "Challengers",
        "Warriors",
        "bat",
        "Cricket Stadium",
        "2023-05-15",
        "Warriors won by 5 wickets"
    ))
    
    # Insert a sample innings
    cursor.execute('''
        INSERT INTO innings (match_id, inning_number, batting_team, bowling_team, total_runs, total_wickets_lost, overs_played, raw_total_score_string)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        sample_match_id,
        1,
        "Challengers",
        "Warriors",
        165,
        8,
        20.0,
        "165/8"
    ))
    inning_id = cursor.lastrowid
    
    # Insert sample batting stats
    cursor.execute('''
        INSERT INTO batting_stats (inning_id, match_id, player_name, status, runs, balls, fours, sixes, strike_rate)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        inning_id,
        sample_match_id,
        "John Smith",
        "c Maxwell b Johnson",
        78,
        52,
        6,
        3,
        150.0
    ))
    
    # Insert sample bowling stats
    cursor.execute('''
        INSERT INTO bowling_stats (inning_id, match_id, player_name, overs, maidens, runs_conceded, wickets, economy)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        inning_id,
        sample_match_id,
        "Mike Johnson",
        4.0,
        0,
        32,
        3,
        8.0
    ))
    
    # Insert second innings
    cursor.execute('''
        INSERT INTO innings (match_id, inning_number, batting_team, bowling_team, total_runs, total_wickets_lost, overs_played, raw_total_score_string)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        sample_match_id,
        2,
        "Warriors",
        "Challengers",
        166,
        5,
        19.2,
        "166/5"
    ))
    inning_id = cursor.lastrowid
    
    # Insert sample batting stats for second innings
    cursor.execute('''
        INSERT INTO batting_stats (inning_id, match_id, player_name, status, runs, balls, fours, sixes, strike_rate)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        inning_id,
        sample_match_id,
        "David Warner",
        "not out",
        85,
        60,
        7,
        4,
        141.67
    ))
    
    # Insert sample bowling stats for second innings
    cursor.execute('''
        INSERT INTO bowling_stats (inning_id, match_id, player_name, overs, maidens, runs_conceded, wickets, economy)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        inning_id,
        sample_match_id,
        "James Anderson",
        4.0,
        0,
        38,
        2,
        9.5
    ))
    
    print("Sample match data created successfully.")
    conn.commit()
else:
    print(f"Found {len(json_files)} scorecard files. Processing and inserting into DB...")
    processed_files = 0
    skipped_files = 0

    for filepath in json_files:
        filename = os.path.basename(filepath)
        match_id = extract_match_id(filename)

        if match_id is None:
            print(f"Warning: Could not extract match ID from {filename}. Skipping.")
            skipped_files += 1
            continue

        # --- Check if match already exists to avoid duplicates ---
        cursor.execute("SELECT 1 FROM matches WHERE match_id = ?", (match_id,))
        if cursor.fetchone():
            # print(f"Match ID {match_id} ({filename}) already exists in DB. Skipping.")
            skipped_files +=1
            continue # Skip this file

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # --- Insert Match Details ---
            md = data.get('match_details', {})
            teams = md.get('teams', {})
            toss = md.get('toss', {})
            match_info = (
                match_id,
                filename,
                teams.get('team1'),
                teams.get('team2'),
                toss.get('winner'),
                toss.get('decision'),
                md.get('venue'),
                md.get('date'),
                md.get('result'),
            )
            cursor.execute('''
                INSERT INTO matches (match_id, filename, team1, team2, toss_winner, toss_decision, venue, date, result)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', match_info)

            # --- Process Innings ---
            innings_list = data.get('innings', [])
            for i, inning_data in enumerate(innings_list):
                inning_num = i + 1
                batting_team = inning_data.get('batting_team')
                bowling_team = None # Determine bowling team
                if teams.get('team1') and teams.get('team2'):
                    if batting_team == teams['team1']: bowling_team = teams['team2']
                    elif batting_team == teams['team2']: bowling_team = teams['team1']

                raw_score = inning_data.get('total_score', '0/0')
                total_runs, total_wickets = parse_score(raw_score)
                overs_played = parse_overs_float(inning_data.get('overs'))

                # Insert Inning and get its generated ID
                inning_info = (
                    match_id,
                    inning_num,
                    batting_team,
                    bowling_team,
                    total_runs,
                    total_wickets,
                    overs_played,
                    str(raw_score) # Store original string too
                )
                cursor.execute('''
                    INSERT INTO innings (match_id, inning_number, batting_team, bowling_team, total_runs, total_wickets_lost, overs_played, raw_total_score_string)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', inning_info)
                inning_id = cursor.lastrowid # Get the auto-generated ID

                # --- Insert Batting Stats ---
                batting_list = inning_data.get('batting', [])
                for batsman in batting_list:
                    batting_stat = (
                        inning_id,
                        match_id,
                        batsman.get('name'),
                        batsman.get('status'),
                        batsman.get('runs'),
                        batsman.get('balls'),
                        batsman.get('fours'),
                        batsman.get('sixes'),
                        batsman.get('strike_rate')
                    )
                    cursor.execute('''
                        INSERT INTO batting_stats (inning_id, match_id, player_name, status, runs, balls, fours, sixes, strike_rate)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', batting_stat)

                # --- Insert Bowling Stats ---
                bowling_list = inning_data.get('bowling', [])
                for bowler in bowling_list:
                    bowling_stat = (
                        inning_id,
                        match_id,
                        bowler.get('name'),
                        parse_overs_float(bowler.get('overs')), # Ensure overs are float
                        bowler.get('maidens'),
                        bowler.get('runs'), # Runs conceded
                        bowler.get('wickets'),
                        bowler.get('economy')
                    )
                    cursor.execute('''
                        INSERT INTO bowling_stats (inning_id, match_id, player_name, overs, maidens, runs_conceded, wickets, economy)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', bowling_stat)

                # --- Insert Fall of Wickets ---
                fow_list = inning_data.get('fall_of_wickets', [])
                for fow in fow_list:
                    fow_stat = (
                        inning_id,
                        match_id,
                        fow.get('score'),
                        fow.get('wicket'),
                        fow.get('player'),
                        parse_overs_float(fow.get('overs')) # Use helper to clean
                    )
                    cursor.execute('''
                        INSERT INTO fall_of_wickets (inning_id, match_id, score_at_fall, wicket_number, player_out, over_at_fall)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', fow_stat)

            processed_files += 1

        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
            skipped_files += 1
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {filename}. It might be corrupted. Skipping.")
            skipped_files += 1
        except sqlite3.Error as e:
             print(f"Database error processing {filename}: {e}. Skipping.")
             # Consider rolling back transaction for this file if needed
             skipped_files += 1
        except Exception as e:
            print(f"An unexpected error occurred while processing {filename}: {e}. Skipping.")
            skipped_files += 1

    # --- Commit Changes and Close Connection ---
    if processed_files > 0:
        print(f"\nCommitting changes to {DB_NAME}...")
        conn.commit()
        print("Changes committed.")
    else:
        print("\nNo new files processed, no changes to commit.")

    print(f"\nProcessing Complete. Processed: {processed_files}, Skipped/Existing: {skipped_files}")

conn.close()
print("Database connection closed.")
print(f"\nDatabase '{DB_NAME}' has been created and is ready to use with gemini.py")
