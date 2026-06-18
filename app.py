# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import load_and_prepare_data, train_prediction_models, save_data

# 1. Setup the browser tab configuration
st.set_page_config(
    page_title="FIFA Match Predictor", page_icon="icon.jpg", layout="centered"
)

# ==========================================
# 🏆 APP MAIN INTERFACE 
# ==========================================

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
# TAB 1: PREDICTOR (Saves Predictions First)
# ==========================================
with tab1:
    st.header("Predict a Matchup")
    
    available_teams = sorted(pd.concat([df_scores["Team_A_Name"], df_scores["Team_B_Name"]]).dropna().unique())
    is_custom_match = st.checkbox("➕ Predict unlisted match.")
    
    if is_custom_match:
        team_a = st.selectbox("Select Team A:", available_teams, key="custom_team_a_select")
        team_b = st.selectbox("Select Team B:", available_teams, key="custom_team_b_select")
        
        match_round = st.selectbox(
            "Select Tournament Round:", 
            ["Round of 32","Round of 16","Quarter-Finals", "Semi-Finals", "Third-Place Playoff", "Final"]
        )
        
        rank_a = st.number_input("Enter Rank for Team A:", value=10.0, step=1.0, key="custom_rank_a")
        rank_b = st.number_input("Enter Rank for Team B:", value=10.0, step=1.0, key="custom_rank_b")
        missing_data = False
    else:
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

    if not missing_data:
        if st.button("Generate Prediction", type="primary", key="main_prediction_btn"):
            if team_a == team_b:
                st.error("❌ A team cannot play against itself.")
                st.stop()

            # Calculate predictions immediately
            rank_diff = abs(rank_a - rank_b)
            unplayed_match = pd.DataFrame(
                [[rank_a, rank_b, rank_diff, team_a, team_b]], 
                columns=["prior_to_game_Rank_Team_A", "prior_to_game_Rank_Team_B", "Rank_diff", "Team_A_Name", "Team_B_Name"]
            )
            
            model_A, model_B = train_prediction_models(df_scores)
            pred_A = max(0, round(model_A.predict(unplayed_match)[0]))
            pred_B = max(0, round(model_B.predict(unplayed_match)[0]))

            # Save prediction directly to the database row
            if is_custom_match:
                same_round_games = df_scores[df_scores["Round"].astype(str) == str(match_round)]
                game_num = 1 if same_round_games.empty else int(same_round_games["Game"].max()) + 1
                
                new_row = {
                    "Round": match_round,
                    "Game": game_num,
                    "Match": f"{team_a}_vs_{team_b}",
                    "Score_Team_A": None,
                    "Score_Team_B": None,
                    "prior_to_game_Rank_Team_A": float(rank_a),
                    "prior_to_game_Rank_Team_B": float(rank_b),
                    "Pred_Team_A": int(pred_A),
                    "Pred_Team_B": int(pred_B)
                }
                df_scores = pd.concat([df_scores, pd.DataFrame([new_row])], ignore_index=True)
            else:
                df_scores.loc[row_idx, "Pred_Team_A"] = int(pred_A)
                df_scores.loc[row_idx, "Pred_Team_B"] = int(pred_B)
            
            save_data(df_scores)
            
            st.success("🎯 Prediction calculated and locked into Google Sheets database!")
            st.markdown("---")
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric(label=f"{team_a} (Rank {int(rank_a)})", value=f"{pred_A} goals")
            with res_col2:
                st.metric(label=f"{team_b} (Rank {int(rank_b)})", value=f"{pred_B} goals")
            st.markdown("---")

# ==========================================
# TAB 2: UPDATER (Restored to independent dropdowns)
# ==========================================
with tab2:
    st.header("Log Game Outcome")
    
    # Independent dropdowns restored exactly like before
    u_team_a = st.selectbox("Team A Who Played:", available_teams, key="ut_a")
    u_team_b = st.selectbox("Team B Who Played:", available_teams, key="ut_b")

    score_a = st.number_input(f"{u_team_a} Final Score:", min_value=0, step=1, value=0, key="score_a_input")
    score_b = st.number_input(f"{u_team_b} Final Score:", min_value=0, step=1, value=0, key="score_b_input")
    
    new_rank_a = st.text_input(f"Update rank for {u_team_a} (Optional):", key="u_rank_a").strip()
    new_rank_b = st.text_input(f"Update rank for {u_team_b} (Optional):", key="u_rank_b").strip()
    
    if st.button("Save Game Result"):
        # Locate the row for the selected matchup
        match_mask = (df_scores["Team_A_Name"] == u_team_a) & (df_scores["Team_B_Name"] == u_team_b)
        
        if not df_scores[match_mask].empty:
            row_index = df_scores[match_mask].index[0]
            
            # Save final scores (preserving the Pred_Team_A/B already stored there from Tab 1)
            df_scores.loc[row_index, "Score_Team_A"] = score_a
            df_scores.loc[row_index, "Score_Team_B"] = score_b
            
            if new_rank_a:
                df_scores.loc[row_index, "prior_to_game_Rank_Team_A"] = float(new_rank_a)
            if new_rank_b:
                df_scores.loc[row_index, "prior_to_game_Rank_Team_B"] = float(new_rank_b)
                
            save_data(df_scores)
            st.success(f"💾 Results Saved! {u_team_a} {score_a} - {score_b} {u_team_b}")
            st.rerun()
        else:
            st.error(f"❌ Matchup '{u_team_a} vs {u_team_b}' not found in your schedule database. Generate a prediction for it in Tab 1 first to register the game!")

# ==========================================
# TAB 3: ANALYTICS (Processes Saved History)
# ==========================================
with tab3:
    st.header("Tournament Insights")
    
    # Filter only games that have been completed/played AND have a prediction score saved
    plot_df = df_scores[df_scores["Score_Team_A"].notna() & df_scores["Pred_Team_A"].notna()].copy()
    
    if plot_df.empty:
        st.warning("No completed matches with saved predictions recorded yet.")
    else:
        all_rounds = ["All Rounds"] + sorted(list(plot_df["Round"].astype(str).unique()))
        selected_round_filter = st.selectbox("🎯 Filter by Round:", all_rounds, key="analytics_round_filter")
        
        if selected_round_filter != "All Rounds":
            plot_df = plot_df[plot_df["Round"].astype(str) == selected_round_filter]
            
        if plot_df.empty:
            st.info("No matches found for the selected round filter.")
            st.stop()

        # Match Performance vs Rank Disparity Scatter Plot
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

        # Combined Goals vs Rank Difference Scatter Plot
        st.markdown("---")
        st.subheader("⚽ Total Match Goals scored vs Rank Disparity")
        
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        plot_df["Combined_Goals"] = plot_df["Score_Team_A"] + plot_df["Score_Team_B"]
        
        ax3.scatter(
            plot_df["Combined_Goals"],
            plot_df["Rank_diff"],
            color="#6f42c1", 
            s=120,
            edgecolors="#4a154b",
            linewidth=1.2,
            alpha=0.85,
            zorder=3
        )
        
        for rank, goals, match in zip(plot_df["Combined_Goals"], plot_df["Rank_diff"], plot_df["Match"]):
            ax3.text(
                rank + 0.5, 
                goals, 
                f" {match} ", 
                fontsize=9, 
                verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.15", fc="#ffffff", ec="#e0e0e0", lw=0.7, alpha=0.8)
            )
            
        ax3.set_xlabel("Combined Goals Scored", fontsize=12, fontweight="bold", labelpad=10)
        ax3.set_ylabel("Rank Difference", fontsize=12, fontweight="bold", labelpad=10)
        ax3.grid(True, linestyle=":", alpha=0.6, color="#cccccc")
        st.pyplot(fig3)

        # Model Accuracy Calculations using frozen history
        st.markdown("---")
        st.subheader("🔮 Model Accuracy: Predicted vs Actual Scores")
        
        plot_df["Error_A"] = abs(plot_df["Score_Team_A"] - plot_df["Pred_Team_A"])
        plot_df["Error_B"] = abs(plot_df["Score_Team_B"] - plot_df["Pred_Team_B"])
        plot_df["Total_Absolute_Error"] = plot_df["Error_A"] + plot_df["Error_B"]
        
        avg_mae = plot_df["Total_Absolute_Error"].mean()
        exact_match_pct = (plot_df["Total_Absolute_Error"] == 0).sum() / len(plot_df) * 100
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Total Matches Evaluated", f"{len(plot_df)}")
        with metric_col2:
            st.metric("Average Error (MAE)", f"{avg_mae:.2f} goals", help="Lower is better.")
        with metric_col3:
            st.metric("Exact Match Accuracy", f"{exact_match_pct:.1f}%")

        # Outcome vs Upset Tracking Table
        chart_data = []
        for _, row in plot_df.iterrows():
            if row["Score_Team_A"] > row["Score_Team_B"]:
                actual_winner = row["Team_A_Name"]
            elif row["Score_Team_B"] > row["Score_Team_A"]:
                actual_winner = row["Team_B_Name"]
            else:
                actual_winner = "Draw"
                
            if row["Pred_Team_A"] > row["Pred_Team_B"]:
                predicted_winner = row["Team_A_Name"]
            elif row["Pred_Team_B"] > row["Pred_Team_A"]:
                predicted_winner = row["Team_B_Name"]
            else:
                predicted_winner = "Draw"
                
            rank_a = row["prior_to_game_Rank_Team_A"]
            rank_b = row["prior_to_game_Rank_Team_B"]
            
            is_upset = False
            if actual_winner == row["Team_A_Name"] and rank_a > rank_b:
                is_upset = True
            elif actual_winner == row["Team_B_Name"] and rank_b > rank_a:
                is_upset = True

            chart_data.append({
                "Matchup": f"{row['Match']}",
                "Actual Score": f"{int(row['Score_Team_A'])} - {int(row['Score_Team_B'])}",
                "Predicted Score": f"{int(row['Pred_Team_A'])} - {int(row['Pred_Team_B'])}",
                "Predicted Winner": predicted_winner,
                "Actual Winner": actual_winner,
                "Rank Difference": row["Rank_diff"],
                "Is_Upset": is_upset
            })
        
        summary_df = pd.DataFrame(chart_data)
        
        st.subheader("📝 Outcome Analysis")
        st.caption("🟧 = Lower ranked team won")
        st.caption("🟦 = Draw with Rank Diff > 40")
        
        columns_to_show = ["Matchup", "Actual Score", "Predicted Score", "Predicted Winner", "Actual Winner", "Rank Difference"]
        visible_df = summary_df[columns_to_show].copy()

        def highlight_match_rows(row):
            styles = [''] * len(row)
            original_row = summary_df[summary_df["Matchup"] == row["Matchup"]].iloc[0]
            
            if original_row["Is_Upset"]:
                return ['background-color: #ffe8cc; color: #cc6600; font-weight: bold;'] * len(row)
            if original_row["Actual Winner"] == "Draw" and original_row["Rank Difference"] > 40:
                return ['background-color: #d0ebff; color: #0066cc; font-weight: bold;'] * len(row)
            return styles

        styled_df = visible_df.style.apply(highlight_match_rows, axis=1)
        st.dataframe(styled_df, width="stretch", hide_index=True)

        # Prediction Accuracy Breakdown Badges
        st.markdown("### 🏆 Prediction Accuracy Breakdown")
        
        correct_winners = 0
        predicted_draw_but_someone_won = 0
        predicted_winner_but_was_draw = 0
        
        for _, row in summary_df.iterrows():
            p_win = row["Predicted Winner"]
            a_win = row["Actual Winner"]
            
            if p_win == a_win:
                correct_winners += 1
            elif p_win == "Draw" and a_win != "Draw":
                predicted_draw_but_someone_won += 1
            elif p_win != "Draw" and a_win == "Draw":
                predicted_winner_but_was_draw += 1

        badge_col1, badge_col2, badge_col3 = st.columns(3)
        with badge_col1:
            st.metric(label="🎯 Correct Outcomes", value=correct_winners)
        with badge_col2:
            st.metric(label="⏳ Missed Draws", value=predicted_draw_but_someone_won)
        with badge_col3:
            st.metric(label="💔 Broken Deadlocks", value=predicted_winner_but_was_draw)
            
# ==========================================
        # 📊 TRUE SCROLLABLE VERTICAL BAR CHART
        # ==========================================
        st.markdown("---")
        st.subheader("Goal Prediction Comparison")
        st.caption("🟩 = Actual Combined Goals")
        st.caption("🟧 = Predicted Combined Goals")
        
        # 1. Use ALL games instead of trimming to 15
        display_df = summary_df.copy()
        num_matches = len(display_df)
        
        # 2. Dynamically set the width so the bars maintain their exact original size
        # Each match gets 0.6 inches. If there are many matches, it becomes very wide.
        base_width = 6
        chart_width = max(base_width, num_matches * 0.35)
        
        # Structure, heights, and style remain exactly the same as your original
        fig2, ax2 = plt.subplots(figsize=(chart_width, 2.2))
        fig2.subplots_adjust(bottom=0.42)
        
        x_indices = np.arange(num_matches)
        bar_width = 0.35
        
        # Parse totals back out for the bar chart
        actual_totals = display_df["Actual Score"].apply(lambda s: sum(map(int, s.split(" - "))))
        predicted_totals = display_df["Predicted Score"].apply(lambda s: sum(map(int, s.split(" - "))))
        max_goals = int(max(actual_totals.max(), predicted_totals.max())) + 1
        ax2.set_yticks(range(0, max_goals))
        # Your original vertical bars preserved exactly
        ax2.bar(x_indices - bar_width/2, actual_totals, bar_width, label="Actual Combined Goals", color="#28a745", alpha=0.85)
        ax2.bar(x_indices + bar_width/2, predicted_totals, bar_width, label="Predicted Combined Goals", color="#fd7e14", alpha=0.85)
        
        ax2.set_ylabel("Total Match Goals", fontsize=8, fontweight="bold")

        ax2.set_xticks(x_indices)
        ax2.set_xticklabels(display_df["Matchup"], rotation=45, ha="right", fontsize=6)
        ax2.grid(axis='y', linestyle=":", alpha=0.5)
        
        # Adjust layout tightly to make sure labels aren't cut off
        plt.tight_layout()
        
        # 3. Create a clean horizontal scroll window using a Streamlit container with custom HTML
        # This keeps the layout container at 100% page width, but allows the wide chart inside to scroll sideways!
        import io
        import base64
        
        # Save plot to a buffer to embed as an image string
        buf = io.BytesIO()
        fig2.savefig(buf, format="png", bbox_inches="tight",dpi=120)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig2)  # Clean up memory
        
        # Display via scrollable HTML div element
        html_code = f"""
        <div style="overflow-x: auto; overflow-y: hidden; white-space: nowrap; width: 100%; border: 1px solid #e6e9ef; padding: 10px; border-radius: 5px;">
            <img src="data:image/png;base64,{img_base64}" style="max-width: none; height: auto;"/>
        </div>
        """
        st.components.v1.html(html_code, height=420, scrolling=False)