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
    
    available_teams = sorted(pd.concat([df_scores["Team_A_Name"], df_scores["Team_B_Name"]]).dropna().unique())
    
    is_custom_match = st.checkbox("➕ Predict unlisted match.")
    
    if is_custom_match:
        # Custom Mode: Use dropdown selections for teams
        team_a = st.selectbox("Select Team A:", available_teams, key="custom_team_a_select")
        team_b = st.selectbox("Select Team B:", available_teams, key="custom_team_b_select")
        
        # Select box for the tournament stage round
        match_round = st.selectbox(
            "Select Tournament Round:", 
            ["Round of 32","Round of 16","Quarter-Finals", "Semi-Finals", "Third-Place Playoff", "Final"]
        )
        
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
                    save_data(df_scores)
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
            if team_a == team_b:
                st.error("❌ A team cannot play against itself.")
                st.stop()
                
            # If a custom match was generated, save it to the CSV structure as an unplayed placeholder
            if is_custom_match:
                existing_match = df_scores[(df_scores["Team_A_Name"] == team_a) & (df_scores["Team_B_Name"] == team_b) & (df_scores["Round"].astype(str) == str(match_round))]
                
                if existing_match.empty:
                    same_round_games = df_scores[df_scores["Round"].astype(str) == str(match_round)]
                    game_num = 1 if same_round_games.empty else int(same_round_games["Game"].max()) + 1
                    
                    new_row = {
                        "Round": match_round,
                        "Game": game_num,
                        "Match": f"{team_a}_vs_{team_b}",
                        "Score_Team_A": None,
                        "Score_Team_B": None,
                        "prior_to_game_Rank_Team_A": float(rank_a),
                        "prior_to_game_Rank_Team_B": float(rank_b)
                    }
                    df_scores = pd.concat([df_scores, pd.DataFrame([new_row])], ignore_index=True)
                    
                    # Clean internal memory headers and write back to scores.csv file
                    # cols_to_drop = ['Score_diff', 'Rank_diff', 'Team_A_Name', 'Team_B_Name']
                    # df_to_save = df_scores.drop(columns=[c for c in cols_to_drop if c in df_scores.columns])
                    # df_to_save.to_csv("scores.csv", index=False)
                    # st.info(f"✨ Matchup registered into the tournament schedule: {team_a} vs {team_b} ({match_round})")
                    save_data(df_scores)

            rank_diff = abs(rank_a - rank_b)
            
            unplayed_match = pd.DataFrame(
                [[rank_a, rank_b, rank_diff, team_a, team_b]], 
                columns=["prior_to_game_Rank_Team_A", "prior_to_game_Rank_Team_B", "Rank_diff", "Team_A_Name", "Team_B_Name"]
            )
            
            model_A, model_B = train_prediction_models(df_scores)
            
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
    
    new_rank_a = st.text_input(f"New rank for {u_team_a} (Leave blank to keep current/default):", key="rank_a_input").strip()
    new_rank_b = st.text_input(f"New rank for {u_team_b} (Leave blank to keep current/default):", key="rank_b_input").strip()
    
    if st.button("Save Game"):
        match_mask = (df_scores["Team_A_Name"] == u_team_a) & (df_scores["Team_B_Name"] == u_team_b)
        
        if not df_scores[match_mask].empty:
            row_index = df_scores[match_mask].index[0]
            df_scores.loc[row_index, "Score_Team_A"] = score_a
            df_scores.loc[row_index, "Score_Team_B"] = score_b
            
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
        else:
            st.error("❌ Matchup not found in your schedule database. Check the predictor tab to create it first!")
            st.stop()
            
        save_data(df_scores)
        
        st.success(f"💾 Results Saved! {u_team_a} {score_a} - {score_b} {u_team_b}")
        st.rerun()
# ==========================================
# TAB 3: ANALYTICS
# ==========================================
with tab3:
    st.header("Historical Tournament Insights")
    
    # Filter only games that have been completed/played
    plot_df = df_scores[df_scores["Score_Team_A"].notna()].copy()
    
    if plot_df.empty:
        st.warning("No played matches recorded yet to display analytics charts.")
    else:
        # 1. NEW: Round Filter Sidebar/Dropdown for Analytics Tab
        all_rounds = ["All Rounds"] + sorted(list(plot_df["Round"].astype(str).unique()))
        selected_round_filter = st.selectbox("🎯 Filter by Round:", all_rounds, key="analytics_round_filter")
        
        # Apply the filter if something specific is chosen
        if selected_round_filter != "All Rounds":
            plot_df = plot_df[plot_df["Round"].astype(str) == selected_round_filter]
            
        if plot_df.empty:
            st.info("No matches found for the selected round filter.")
            st.stop()

        # 2. YOUR ORIGINAL VISUALIZATION (Preserved exactly)
        st.subheader("Match Performance vs Rank Disparity")
        fig, ax = plt.subplots(figsize=(8, 5))
        
        ax.scatter(
            plot_df["Score_diff"], 
            plot_df["Rank_diff"], 
            color="#007bff", 
            s=120, 
            edgecolors="#004085", 
            linewidth=1.2, 
            alpha=0.85, 
            zorder=3,
            label="Played Matches"
        )
        
        import numpy as np
        
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

        ax.set_xlabel("Score Difference", fontsize=12, fontweight="bold", labelpad=10, color="#1a1a1a")
        ax.set_ylabel("Rank Difference", fontsize=12, fontweight="bold", labelpad=10, color="#1a1a1a")
        ax.grid(True, linestyle=":", alpha=0.6, color="#cccccc")
        st.pyplot(fig)

        # 3. NEW VISUALIZATION: Predicted vs Actual Scores
        st.markdown("---")
        st.subheader("🔮 Model Accuracy: Predicted vs Actual Scores")
        
        # Generate predictions for the data currently in view using your existing model runner
        model_A, model_B = train_prediction_models(df_scores)
        
        # Build features dataframe matching what CatBoost expects
        X_eval = plot_df[["prior_to_game_Rank_Team_A", "prior_to_game_Rank_Team_B", "Rank_diff", "Team_A_Name", "Team_B_Name"]]
        
        # Generate predictions and round them down to logical football goals (>= 0)
        plot_df["Pred_Team_A"] = np.clip(np.round(model_A.predict(X_eval)), 0, None)
        plot_df["Pred_Team_B"] = np.clip(np.round(model_B.predict(X_eval)), 0, None)
        
        # Calculate Error Metrics per game
        plot_df["Error_A"] = abs(plot_df["Score_Team_A"] - plot_df["Pred_Team_A"])
        plot_df["Error_B"] = abs(plot_df["Score_Team_B"] - plot_df["Pred_Team_B"])
        plot_df["Total_Absolute_Error"] = plot_df["Error_A"] + plot_df["Error_B"]
        
        # Display aggregate KPIs for the current selection
        avg_mae = plot_df["Total_Absolute_Error"].mean()
        exact_match_pct = (plot_df["Total_Absolute_Error"] == 0).sum() / len(plot_df) * 100
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Total Matches Evaluated", f"{len(plot_df)}")
        with metric_col2:
            st.metric("Average Error (MAE)", f"{avg_mae:.2f} goals", help="Lower is better. 0.00 means perfectly accurate scorelines.")
        with metric_col3:
            st.metric("Exact Match Accuracy", f"{exact_match_pct:.1f}%", help="Percentage of games where both team scores were predicted perfectly.")

        # Create interactive bar chart comparison using Streamlit's native layout options
        # We transform the data into a clean viewable summary table
        chart_data = []
        for _, row in plot_df.iterrows():
            chart_data.append({
                "Matchup": f"{row['Match']} ({row['Round']})",
                "Actual A": row["Score_Team_A"],
                "Predicted A": row["Pred_Team_A"],
                "Actual B": row["Score_Team_B"],
                "Predicted B": row["Pred_Team_B"],
                "Total Deviation": int(row["Total_Absolute_Error"])
            })
        
        summary_df = pd.DataFrame(chart_data)
        
        # Display an expandable breakdown layout for details per game
        with st.expander("📊 View Detailed Accuracy Logs Per Game"):
            st.dataframe(
                summary_df.set_index("Matchup"),
                use_container_width=True
            )
            
        # Draw a grouped bar chart for visual score distribution comparisons
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        
        # Sample limit the bar plot visualization if too dense, to keep it highly readable
        display_df = summary_df.tail(15) if len(summary_df) > 15 else summary_df
        
        x_indices = np.arange(len(display_df))
        bar_width = 0.35
        
        # Sum goals for simpler visual trend tracking (Actual combined vs Predicted combined)
        actual_totals = display_df["Actual A"] + display_df["Actual B"]
        predicted_totals = display_df["Predicted A"] + display_df["Predicted B"]
        
        ax2.bar(x_indices - bar_width/2, actual_totals, bar_width, label="Actual Combined Goals", color="#28a745", alpha=0.85)
        ax2.bar(x_indices + bar_width/2, predicted_totals, bar_width, label="Predicted Combined Goals", color="#fd7e14", alpha=0.85)
        
        ax2.set_ylabel("Total Match Goals", fontsize=11, fontweight="bold")
        ax2.set_title("Goal Production Comparison", fontsize=13, fontweight="bold", pad=15)
        ax2.set_xticks(x_indices)
        ax2.set_xticklabels(display_df["Matchup"], rotation=45, ha="right", fontsize=9)
        ax2.legend()
        ax2.grid(axis='y', linestyle=":", alpha=0.5)
        
        st.pyplot(fig2)
