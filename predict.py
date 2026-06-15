# predict.py
import pandas as pd
from utils import load_and_prepare_data, train_prediction_models


def predict_matchup(team_a_input, team_b_input, df_scores, model_A, model_B):
    # Find the unplayed game row
    match_row = df_scores[
        (df_scores["Team_A_Name"] == team_a_input)
        & (df_scores["Team_B_Name"] == team_b_input)
    ]

    if match_row.empty:
        print(f"\n❌ Matchup '{team_a_input} vs {team_b_input}' not found in the dataset.")
        print("Note: Ensure spelling/casing matches your dataset exactly.")
        return

    # --- NEW: DYNAMIC RANK CHECKING & FALLBACK INPUTS ---
    new_team_a_rank = match_row["prior_to_game_Rank_Team_A"].iloc[0]
    new_team_b_rank = match_row["prior_to_game_Rank_Team_B"].iloc[0]

    # If Team A's rank is null/missing
    if pd.isna(new_team_a_rank):
        print(f"⚠️  Rank for {team_a_input} is missing in the dataset.")
        while True:
            try:
                new_team_a_rank = float(input(f"Enter the rank for {team_a_input}: ").strip())
                break
            except ValueError:
                print("❌ Invalid input. Please enter a number.")

    # If Team B's rank is null/missing
    if pd.isna(new_team_b_rank):
        print(f"⚠️  Rank for {team_b_input} is missing in the dataset.")
        while True:
            try:
                new_team_b_rank = float(input(f"Enter the rank for {team_b_input}: ").strip())
                break
            except ValueError:
                print("❌ Invalid input. Please enter a number.")

    # Calculate rank difference dynamically based on what we now have
    new_rank_diff = abs(new_team_a_rank - new_team_b_rank)
    # -----------------------------------------------------

    # Format data for the model
    unplayed_match = pd.DataFrame(
        [[new_team_a_rank, new_team_b_rank, new_rank_diff]],
        columns=[
            "prior_to_game_Rank_Team_A",
            "prior_to_game_Rank_Team_B",
            "Rank_diff",
        ],
    )

    # Predict scores
    predicted_score_A = round(model_A.predict(unplayed_match)[0])
    predicted_score_B = round(model_B.predict(unplayed_match)[0])

    print("\n=== MATCH PREDICTION ===")
    print(f"{team_a_input} (Rank {int(new_team_a_rank)}): {predicted_score_A:d} points")
    print(f"{team_b_input} (Rank {int(new_team_b_rank)}): {predicted_score_B:d} points")
    print("=========================")


if __name__ == "__main__":
    # 1. Load data
    df_scores = load_and_prepare_data("scores.csv")

    # 2. Collect interactive inputs directly in the terminal
    print("\n🔮 --- FIFA Match Predictor ---")
    team_a = input("Enter Team A name/code: ").strip()
    team_b = input("Enter Team B name/code: ").strip()

    if not team_a or not team_b:
        print("❌ Error: Both team names must be provided.")
        exit(1)

    # 3. Train models
    model_A, model_B = train_prediction_models(df_scores)

    # 4. Generate the prediction
    predict_matchup(team_a, team_b, df_scores, model_A, model_B)
