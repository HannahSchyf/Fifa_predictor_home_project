# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_and_prepare_data, train_prediction_models, save_data

# 1. Setup the browser tab configuration
st.set_page_config(
    page_title="FIFA Match Predictor", page_icon="icon.jpg", layout="centered"
)

# ==========================================
# 🏆 APP MAIN INTERFACE 
# ==========================================

# Create side-by-side columns for the JPG and Title text
col1, col2 = st.columns([1, 4])
with col1:
    st.image("ball.jpg", width=75)
with col2:
    st.title("FIFA Match Predictor & Updater")

st.write("Predict upcoming tournament match scores.")
st.markdown("---")

# Load data fresh on every page load
df_scores = load_and_prepare_data("scores.csv")

# Create clean tabs for mobile switching
tab1, tab2, tab3 = st.tabs(["🔮 Predict Match", "📝 Update Result", "📊 View Analytics"])
# ==========================================
# TAB 1: PREDICTOR
# ==========================================
with tab1:
    st.header("Predict a Matchup")
    
    # Extract unique teams dynamically
    available_teams = sorted(pd.concat([df_scores["Team_A_Name"], df_scores["Team_B_Name"]]).dropna().unique())
    
    # NEW: Toggle to allow predicting unlisted/custom/TBC matches
    is_custom_match = st.checkbox("➕ Predict a custom/unlisted match (e.g. TBC Finals)")
    
    if is_custom_match:
        # Custom Mode: Let users enter anything manually
        team_a = st.text_input("Enter Team A Name:", placeholder="e.g., France")
        team_b = st.text_input("Enter Team B Name:", placeholder="e.g., Argentina")
        rank_a = st.number_input("Enter Rank for Team A:", value=10.0, step=1.0, key="custom_rank_a")
        rank_b = st.number_input("Enter Rank for Team B:", value=10.0, step=1.0, key="custom_rank_b")
        missing_data = False
    else:
        # Standard Mode: Read from your CSV schedule dropdowns
        team_a = st.selectbox("Select Team A:", available_teams, index=0)
        team_b = st.selectbox("Select Team B:", available_teams, index=1)
        
        match_row = df_scores[(df_scores["Team_A_Name"] == team_a) & (df_scores["Team_B_Name"] == team_b)]
        
        if match_row.empty:
            st.error(f"❌ Matchup '{team_a} vs {team_b}' not found in schedule. Check the custom box above to predict it!")
            missing_data = True
        else:
            row_idx = match_row.index[0]
            csv_rank_a = df_scores.loc[row_idx, "prior_to_game_Rank_Team_A"]
            csv_rank_b = df_scores.loc[row_idx, "prior_to_game_Rank_Team_B"]
            missing_data = False
            
            if pd.isna(csv_rank_a):
                missing_data = True
                new_a = st.number_input(f"⚠️ Rank for {team_a} missing. Enter to save:", value=10.0, key=f"fb_rank_{team_a}")
                if st.button(f"💾 Save Rank for {team_a}", key=f"save_btn_{team_a}"):
                    df_scores.loc[row_idx, "prior_to_game_Rank_Team_A"] = float(new_a)
                    save_data(df_scores) # Using your utility save function
                    st.success(f"Saved {team_a} rank!")
                    st.rerun()
            else:
                rank_a = csv_rank_a
                
            if pd.isna(csv_rank_b):
                missing_data = True
                new_b = st.number_input(f"⚠️ Rank for {team_b} missing. Enter to save:", value=10.0, key=f"fb_rank_{team_b}")
                if st.button(f"💾 Save Rank for {team_b}", key=f"save_btn_{team_b}"):
                    df_scores.loc[row_idx, "prior_to_game_Rank_Team_B"] = float(new_b)
                    save_data(df_scores)
                    st.success(f"Saved {team_b} rank!")
                    st.rerun()
            else:
                rank_b = csv_rank_b

    # Prediction Engine Execution block
    if not missing_data:
        if st.button("Generate Prediction", type="primary", key="main_prediction_btn"):
            rank_diff = abs(rank_a - rank_b)
            
            # Formulated exactly what CatBoost expects (including string team names)
            unplayed_match = pd.DataFrame(
                [[rank_a, rank_b, rank_diff, team_a, team_b]], 
                columns=["prior_to_game_Rank_Team_A", "prior_to_game_Rank_Team_B", "Rank_diff", "Team_A_Name", "Team_B_Name"]
            )
            
            model_A, model_B = train_prediction_models(df_scores)
            
            # Predict scores, ensuring no negative outputs
            pred_A = max(0, round(model_A.predict(unplayed_match)[0]))
            pred_B = max(0, round(model_B.predict(unplayed_match)[0]))
            
            st.markdown("---")
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric(label=f"{team_a} (Rank {int(rank_a)})", value=f"{pred_A} goals")
            with res_col2:
                st.metric(label=f"{team_b} (Rank {int(rank_b)})", value=f"{pred_B} goals")
            st.markdown("---")

# ==========================================
# TAB 2: UPDATER
# ==========================================
with tab2:
    st.header("Log Game Outcome")
    
    u_team_a = st.selectbox("Team A Who Played:", available_teams, key="ut_a")
    u_team_b = st.selectbox("Team B Who Played:", available_teams, key="ut_b")
    
    score_a = st.number_input(f"{u_team_a} Final Score:", min_value=0, step=1, value=0, key="score_a_input")
    score_b = st.number_input(f"{u_team_b} Final Score:", min_value=0, step=1, value=0, key="score_b_input")
    
    # Optional rank update inputs
    new_rank_a = st.text_input(f"New rank for {u_team_a} (Leave blank to keep current):", key="rank_a_input").strip()
    new_rank_b = st.text_input(f"New rank for {u_team_b} (Leave blank to keep current):", key="rank_b_input").strip()
    
    if st.button("Save Game to CSV"):
        match_mask = (df_scores["Team_A_Name"] == u_team_a) & (df_scores["Team_B_Name"] == u_team_b)
        
        if not df_scores[match_mask].empty:
            row_index = df_scores[match_mask].index[0]
            
            # Log the fresh scores
            df_scores.loc[row_index, "Score_Team_A"] = score_a
            df_scores.loc[row_index, "Score_Team_B"] = score_b
            
            # Process potential rank changes or flag if data is missing completely
            if new_rank_a:
                df_scores.loc[row_index, "prior_to_game_Rank_Team_A"] = float(new_rank_a)
            elif pd.isna(df_scores.loc[row_index, "prior_to_game_Rank_Team_A"]):
                st.error(f"❌ Rank for {u_team_a} is missing in CSV. You must type a rank to save.")
                st.stop()
                
            if new_rank_b:
                df_scores.loc[row_index, "prior_to_game_Rank_Team_B"] = float(new_rank_b)
            elif pd.isna(df_scores.loc[row_index, "prior_to_game_Rank_Team_B"]):
                st.error(f"❌ Rank for {u_team_b} is missing in CSV. You must type a rank to save.")
                st.stop()
            
            # Clean structure and write changes back to local scores.csv
            cols_to_drop = ['Score_diff', 'Rank_diff', 'Team_A_Name', 'Team_B_Name']
            df_to_save = df_scores.drop(columns=[c for c in cols_to_drop if c in df_scores.columns])
            df_to_save.to_csv("scores.csv", index=False)
            
            st.success(f"💾 Results Saved! {u_team_a} {score_a} - {score_b} {u_team_b}.")
            st.rerun()
        else:
            st.error("❌ Matchup not found in your schedule database.")

# ==========================================
# TAB 3: VISUALIZATIONS & PLOTS (CLEAN VERSION)
# ==========================================
with tab3:
    st.header("📊 Historical Performance Analytics")
    st.write("Visual relationship between score margins and pre-game ranking differences.")

    # 1. Drop missing values safely
    plot_df = df_scores.dropna(subset=["Score_diff", "Rank_diff"])

    if plot_df.empty:
        st.info("💡 Not enough played match data to display the analytics yet.")
    else:
        # 2. Modern Design Init
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        
        fig, ax = plt.subplots(figsize=(10, 6.5))
        fig.patch.set_facecolor('#ffffff')  # Matches clean web backgrounds
        ax.set_facecolor('#f8f9fa')         # Light grey canvas background

        # 3. Dynamic Color Mapping: Color dots by match outcome
        colors = ['#1f77b4' if x > 0 else '#fc4f30' for x in plot_df["Score_diff"]]

        # 4. Draw the base scatter plot with shadows and larger markers
        scatter = ax.scatter(
            plot_df["Score_diff"], 
            plot_df["Rank_diff"], 
            c=colors, 
            edgecolors="#2b2b2b", 
            s=120,                  
            alpha=0.85, 
            linewidths=1.2,
            zorder=3                
        )

        # 5. Add a bold center line at 0 for Score Difference
        ax.axvline(x=0, color='#6c757d', linestyle='-', linewidth=1.5, alpha=0.7, zorder=2)
        
        # 6. Loop and annotate labels with modern offsets
        for score, rank, match in zip(plot_df["Score_diff"], plot_df["Rank_diff"], plot_df["Match"]):
            offset = 0.18 if score >= 0 else -0.18
            align = 'left' if score >= 0 else 'right'
            
            ax.text(
                score + offset, 
                rank, 
                f" {match} ", 
                fontsize=9.5, 
                fontweight="medium",
                color="#212529",
                alpha=0.9, 
                verticalalignment='center',
                horizontalalignment=align,
                bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec="#e0e0e0", lw=0.8, alpha=0.85)
            )

        # 7. Premium Styling & Typography
        ax.set_xlabel("Score Difference", fontsize=12, fontweight="bold", labelpad=10, color="#1a1a1a")
        ax.set_ylabel("Rank Difference", fontsize=12, fontweight="bold", labelpad=10, color="#1a1a1a")
        ax.set_title("Match Performance vs Rank Disparity", fontsize=15, fontweight="bold", pad=20, color="#111111")
        
        # Customize gridlines to look minimal and elegant
        ax.grid(True, linestyle=":", alpha=0.6, color="#cccccc")
        
        # Remove harsh outer box borders
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_visible(False)

        # 8. Render to Streamlit dashboard
        st.pyplot(fig)
