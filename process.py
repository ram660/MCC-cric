import json
import pandas as pd
import os
import glob
import re # For extracting match ID

# --- Configuration ---
# <<< IMPORTANT: Set this to the directory containing your JSON files >>>
JSON_DIR = r'C:\Users\ramma\Downloads\New folder\data' # Make sure this path is correct

# --- Data Storage ---
all_matches_data = []
all_innings_data = []
all_batting_data = []
all_bowling_data = []
all_fow_data = [] # Fall of Wickets

# --- Helper Function to Extract Match ID from Filename ---
def extract_match_id(filename):
    match = re.search(r'Scorecard_(\d+)\.json', filename)
    if match:
        return int(match.group(1))
    return None # Or raise an error if ID is mandatory

# --- Find and Process JSON Files ---
json_files = glob.glob(os.path.join(JSON_DIR, 'Scorecard_*.json'))

if not json_files:
    print(f"Error: No 'Scorecard_*.json' files found in directory: {JSON_DIR}")
    print("Please check the JSON_DIR path.")
else:
    print(f"Found {len(json_files)} scorecard files. Processing...")

    for filepath in json_files:
        filename = os.path.basename(filepath)
        match_id = extract_match_id(filename)
        if match_id is None:
            print(f"Warning: Could not extract match ID from {filename}. Skipping.")
            continue

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 1. Process Match Details
            md = data.get('match_details', {})
            teams = md.get('teams', {})
            toss = md.get('toss', {})
            match_info = {
                'match_id': match_id,
                'filename': filename,
                'team1': teams.get('team1'),
                'team2': teams.get('team2'),
                'toss_winner': toss.get('winner'),
                'toss_decision': toss.get('decision'),
                'venue': md.get('venue'),
                'date': md.get('date'),
                'result': md.get('result'),
            }
            all_matches_data.append(match_info)

            # 2. Process Innings Details
            innings_list = data.get('innings', [])
            for i, inning_data in enumerate(innings_list):
                inning_num = i + 1
                batting_team = inning_data.get('batting_team')

                # Determine bowling team
                bowling_team = None
                if match_info['team1'] and match_info['team2']:
                    if batting_team == match_info['team1']:
                        bowling_team = match_info['team2']
                    elif batting_team == match_info['team2']:
                        bowling_team = match_info['team1']

                # Parse total score (e.g., "116", "211/8")
                total_score_str = str(inning_data.get('total_score', '0/0')) # Ensure it's a string
                total_runs = 0
                total_wickets = 0
                if '/' in total_score_str:
                    parts = total_score_str.split('/')
                    try:
                        total_runs = int(parts[0])
                        total_wickets = int(parts[1])
                    except (ValueError, IndexError):
                         print(f"Warning: Could not parse total_score '{total_score_str}' in {filename}, inning {inning_num}")
                else:
                    try:
                        total_runs = int(total_score_str)
                        # Assume 10 wickets if not specified and score is just a number (might need refinement)
                        # total_wickets = 10 # Or leave as 0 if unsure
                    except ValueError:
                         print(f"Warning: Could not parse total_score '{total_score_str}' in {filename}, inning {inning_num}")


                inning_info = {
                    'match_id': match_id,
                    'inning': inning_num,
                    'batting_team': batting_team,
                    'bowling_team': bowling_team,
                    'total_runs': total_runs,
                    'total_wickets_lost': total_wickets, # Wickets lost by batting team
                    'overs_played': inning_data.get('overs'),
                    'raw_total_score_string': total_score_str # Keep original for reference
                }
                all_innings_data.append(inning_info)

                # 3. Process Batting Stats
                batting_list = inning_data.get('batting', [])
                for batsman in batting_list:
                    batting_stat = {
                        'match_id': match_id,
                        'inning': inning_num,
                        'batting_team': batting_team,
                        'name': batsman.get('name'),
                        'status': batsman.get('status'),
                        'runs': batsman.get('runs'),
                        'balls': batsman.get('balls'),
                        'fours': batsman.get('fours'),
                        'sixes': batsman.get('sixes'),
                        'strike_rate': batsman.get('strike_rate'),
                    }
                    all_batting_data.append(batting_stat)

                # 4. Process Bowling Stats
                bowling_list = inning_data.get('bowling', [])
                for bowler in bowling_list:
                    bowling_stat = {
                        'match_id': match_id,
                        'inning': inning_num,
                        'bowling_team': bowling_team, # Team fielding/bowling
                        'name': bowler.get('name'),
                        'overs': bowler.get('overs'),
                        'maidens': bowler.get('maidens'),
                        'runs_conceded': bowler.get('runs'), # Renamed for clarity
                        'wickets': bowler.get('wickets'),
                        'economy': bowler.get('economy'),
                    }
                    all_bowling_data.append(bowling_stat)

                # 5. Process Fall of Wickets
                fow_list = inning_data.get('fall_of_wickets', [])
                for fow in fow_list:
                     # Clean 'overs' field (e.g., "0.4 ov", "5 ov") -> float
                    over_str = str(fow.get('overs', '0.0')).lower().replace(' ov', '').strip()
                    try:
                        over_float = float(over_str)
                    except ValueError:
                        print(f"Warning: Could not parse FOW overs '{fow.get('overs')}' to float in {filename}, inning {inning_num}. Using 0.0.")
                        over_float = 0.0

                    fow_stat = {
                        'match_id': match_id,
                        'inning': inning_num,
                        'batting_team': batting_team,
                        'score_at_fall': fow.get('score'),
                        'wicket_number': fow.get('wicket'),
                        'player_out': fow.get('player'),
                        'over_at_fall': over_float,
                    }
                    all_fow_data.append(fow_stat)

        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {filename}. It might be corrupted.")
        except Exception as e:
            print(f"An unexpected error occurred while processing {filename}: {e}")

    # --- Create Pandas DataFrames ---
    if all_matches_data:
        df_matches = pd.DataFrame(all_matches_data)
        df_innings = pd.DataFrame(all_innings_data)
        df_batting = pd.DataFrame(all_batting_data)
        df_bowling = pd.DataFrame(all_bowling_data)
        df_fow = pd.DataFrame(all_fow_data)

        print("\n--- DataFrames Created Successfully ---")

        print("\nMatches DataFrame Head:")
        print(df_matches.head())
        print(f"\nTotal Matches Processed: {len(df_matches)}")

        print("\nInnings DataFrame Head:")
        print(df_innings.head())

        print("\nBatting Stats DataFrame Head:")
        print(df_batting.head())

        print("\nBowling Stats DataFrame Head:")
        print(df_bowling.head())

        print("\nFall of Wickets DataFrame Head:")
        print(df_fow.head())

        # --- Optional: Save DataFrames to efficient format (Parquet recommended) ---
        OUTPUT_DIR = './processed_data/'
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df_matches.to_parquet(os.path.join(OUTPUT_DIR, 'matches.parquet'), index=False)
        df_innings.to_parquet(os.path.join(OUTPUT_DIR, 'innings.parquet'), index=False)
        df_batting.to_parquet(os.path.join(OUTPUT_DIR, 'batting.parquet'), index=False)
        df_bowling.to_parquet(os.path.join(OUTPUT_DIR, 'bowling.parquet'), index=False)
        df_fow.to_parquet(os.path.join(OUTPUT_DIR, 'fall_of_wickets.parquet'), index=False)
        print(f"\nDataFrames saved to {OUTPUT_DIR}")

    else:
        print("\nNo data was processed successfully. No DataFrames created.")