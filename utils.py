# utils.py
import pandas as pd
from sklearn.linear_model import LinearRegression


def load_and_prepare_data(csv_path="scores.csv"):
    """Loads the dataset, calculates differences, splits teams, and returns the DataFrame."""
    try:
        # Read the file
        df_scores = pd.read_csv(csv_path)

        # 1. Calculate the score and rank differences
        df_scores["Score_diff"] = abs(
            df_scores["Score_Team_A"] - df_scores["Score_Team_B"]
        )
        df_scores["Rank_diff"] = abs(
            df_scores["prior_to_game_Rank_Team_A"]
            - df_scores["prior_to_game_Rank_Team_B"]
        )

        # 2. Split the Match column into individual team name columns
        df_scores[["Team_A_Name", "Team_B_Name"]] = df_scores[
            "Match"
        ].str.split("_vs_", expand=True)

        # 3. CRITICAL: Pass the engineered DataFrame back to the rest of the project!
        return df_scores

    except FileNotFoundError:
        print(f"❌ Error: Data file '{csv_path}' not found.")
        raise


def train_prediction_models(df_scores):
    """Trains LinearRegression models on played games and returns them."""
    played_games = df_scores[df_scores["Score_Team_A"].notna()]

    X_train = played_games[
        ["prior_to_game_Rank_Team_A", "prior_to_game_Rank_Team_B", "Rank_diff"]
    ]
    y_A = played_games["Score_Team_A"]
    y_B = played_games["Score_Team_B"]

    model_A = LinearRegression().fit(X_train, y_A)
    model_B = LinearRegression().fit(X_train, y_B)
    return model_A, model_B

def save_data(df_scores, csv_path="scores.csv"):
    """Saves the modified DataFrame back to the CSV file, removing calculated columns."""
    # We drop the dynamically generated columns so they don't bloat the raw CSV file
    cols_to_drop = ['Score_diff', 'Rank_diff', 'Team_A_Name', 'Team_B_Name']
    df_to_save = df_scores.drop(columns=[c for c in cols_to_drop if c in df_scores.columns])
    
    df_to_save.to_csv(csv_path, index=False)
    print(f"💾 Successfully saved updates to {csv_path}!")