# utils.py
import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor
from streamlit_gsheets import GSheetsConnection

def load_and_prepare_data(csv_path="scores.csv"):
    """Loads the dataset from Google Sheets, calculates differences, splits teams, and returns the DataFrame."""
    try:
        # Establish native connection to Google Sheets
        conn = st.connection("gsheets", type=GSheetsConnection)
        
        # Read the spreadsheet live from the cloud (ttl=0 ensures no stale caching)
        df_scores = conn.read(ttl=0)

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

        return df_scores

    except Exception as e:
        st.error(f"❌ Error loading cloud data: {e}")
        raise


# def train_prediction_models(df_scores):
#     """Trains LinearRegression models on played games and returns them."""
#     played_games = df_scores[df_scores["Score_Team_A"].notna()]

#     X_train = played_games[
#         ["prior_to_game_Rank_Team_A", "prior_to_game_Rank_Team_B", "Rank_diff"]
#     ]
#     y_A = played_games["Score_Team_A"]
#     y_B = played_games["Score_Team_B"]

#     model_A = LinearRegression().fit(X_train, y_A)
#     model_B = LinearRegression().fit(X_train, y_B)
#     return model_A, model_B

def train_prediction_models(df_scores):
    """Trains CatBoostRegressor models natively handling categorical team names."""
    played_games = df_scores[df_scores["Score_Team_A"].notna()]

    # Included team names as features for CatBoost
    X_train = played_games[
        ["prior_to_game_Rank_Team_A", "prior_to_game_Rank_Team_B", "Rank_diff", "Team_A_Name", "Team_B_Name"]
    ]
    y_A = played_games["Score_Team_A"]
    y_B = played_games["Score_Team_B"]

    # Initialize CatBoost with categorical feature flags
    model_A = CatBoostRegressor(iterations=150, learning_rate=0.05, depth=6, cat_features=["Team_A_Name", "Team_B_Name"], verbose=0)
    model_B = CatBoostRegressor(iterations=150, learning_rate=0.05, depth=6, cat_features=["Team_A_Name", "Team_B_Name"], verbose=0)
    
    model_A.fit(X_train, y_A)
    model_B.fit(X_train, y_B)
    return model_A, model_B

def save_data(df_scores, csv_path="scores.csv"):
    """Overwrites the Google Sheet rows with the modified tournament database."""
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        
        # ONLY drop custom run-time memory metrics. Leave columns like Pred_Team_A / Pred_Team_B untouched!
        cols_to_drop = ['Score_diff', 'Rank_diff', 'Team_A_Name', 'Team_B_Name']
        df_to_save = df_scores.drop(columns=[c for c in cols_to_drop if c in df_scores.columns])
        
        # 1. Update rows in Google Sheets
        conn.update(data=df_to_save)
        
        # 2. Clear Streamlit's data memory so it pulls the new match instantly
        st.cache_data.clear()
        
    except Exception as e:
        st.error(f"❌ Failed to save changes to Cloud: {e}")