# update_game.py
import pandas as pd
from utils import load_and_prepare_data, save_data

if __name__ == "__main__":
    print("\n📝 --- FIFA Match Result Updater ---")
    
    # 1. Load the data
    df_scores = load_and_prepare_data("scores.csv")
    
    # 2. Get the match identifiers
    team_a = input("Enter Team A name/code: ").strip()
    team_b = input("Enter Team B name/code: ").strip()
    
    # 3. Locate the row in the dataframe
    match_mask = (df_scores["Team_A_Name"] == team_a) & (df_scores["Team_B_Name"] == team_b)
    match_row = df_scores[match_mask]
    
    if match_row.empty:
        print(f"❌ Matchup '{team_a} vs {team_b}' not found in the dataset.")
        exit(1)
        
    row_index = match_row.index[0]
    
    print(f"\n📍 Found scheduled match: {team_a} vs {team_b}")
    
    # 4. Prompt for Scores with validation
    while True:
        try:
            score_a = int(input(f"Enter final score for {team_a}: ").strip())
            score_b = int(input(f"Enter final score for {team_b}: ").strip())
            break
        except ValueError:
            print("❌ Invalid input. Scores must be whole numbers.")

    # 5. Handle Ranks (Keep current rank if left blank, or update if provided)
    current_rank_a = df_scores.loc[row_index, "prior_to_game_Rank_Team_A"]
    current_rank_b = df_scores.loc[row_index, "prior_to_game_Rank_Team_B"]
    
    # Display existing rank or flag if it's missing (NaN)
    print(f"\nℹ️  Current logged ranks -> {team_a}: {current_rank_a if not pd.isna(current_rank_a) else 'Missing'}, {team_b}: {current_rank_b if not pd.isna(current_rank_b) else 'Missing'}")
    
    rank_a_input = input(f"Enter new rank for {team_a} (or press Enter to keep current): ").strip()
    rank_b_input = input(f"Enter new rank for {team_b} (or press Enter to keep current): ").strip()
    
    # Assign new values if typed, otherwise preserve or ask if it was completely blank
    if rank_a_input:
        df_scores.loc[row_index, "prior_to_game_Rank_Team_A"] = float(rank_a_input)
    elif pd.isna(current_rank_a):
        # Force an entry if it was missing completely
        df_scores.loc[row_index, "prior_to_game_Rank_Team_A"] = float(input(f"❌ Rank required. Enter rank for {team_a}: "))
        
    if rank_b_input:
        df_scores.loc[row_index, "prior_to_game_Rank_Team_B"] = float(rank_b_input)
    elif pd.isna(current_rank_b):
        df_scores.loc[row_index, "prior_to_game_Rank_Team_B"] = float(input(f"❌ Rank required. Enter rank for {team_b}: "))

    # 6. Apply the scores to transition the game from "unplayed" to "played"
    df_scores.loc[row_index, "Score_Team_A"] = score_a
    df_scores.loc[row_index, "Score_Team_B"] = score_b
    
    # 7. Save structural updates back to disk
    save_data(df_scores, "scores.csv")
    print("✅ Dataset successfully updated!")
