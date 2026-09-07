import os
import pandas as pd
import pulp
import streamlit as st

# ==========================================
# 📐 MATHEMATICAL OPTIMIZER CORE
# ==========================================
def optimize_lineup(df, salary_cap, name_col, pos_col, sal_col, proj_col):
    """
    Operations Research Phase:
    Uses Integer Linear Programming via PuLP to maximize projected points 
    subject to positional roster limits and financial constraints.
    """
    player_vars = pulp.LpVariable.dicts("Player", df.index, cat="Binary")
    flex_vars = pulp.LpVariable.dicts("FlexPlayer", df.index, cat="Binary")
    
    prob = pulp.LpProblem("Dynamic_Lineup_Optimization", pulp.LpMaximize)
    
    # Objective Function: Maximize chosen projection column
    prob += pulp.lpSum([df.loc[i, proj_col] * (player_vars[i] + flex_vars[i]) for i in df.index])
    
    # Financial Constraint: Combined base + flex salaries cannot cross slider value
    prob += pulp.lpSum([df.loc[i, sal_col] * (player_vars[i] + flex_vars[i]) for i in df.index]) <= salary_cap
    
    # Structural Constraint: A player can only be selected ONCE total
    for i in df.index:
        prob += player_vars[i] + flex_vars[i] <= 1
        
    # Positional Requirements (1 QB, 1 RB, 1 WR, 1 TE, plus 1 FLEX)
    prob += pulp.lpSum([player_vars[i] for i in df.index if str(df.loc[i, pos_col]).strip().upper() == 'QB']) == 1
    prob += pulp.lpSum([player_vars[i] for i in df.index if str(df.loc[i, pos_col]).strip().upper() == 'RB']) == 1
    prob += pulp.lpSum([player_vars[i] for i in df.index if str(df.loc[i, pos_col]).strip().upper() == 'WR']) == 1
    prob += pulp.lpSum([player_vars[i] for i in df.index if str(df.loc[i, pos_col]).strip().upper() == 'TE']) == 1
    
    # Flex Eligibility Constraint: Only RB, WR, or TE can fill the 1 Flex slot
    prob += pulp.lpSum([flex_vars[i] for i in df.index if str(df.loc[i, pos_col]).strip().upper() in ['RB', 'WR', 'TE']]) == 1
    prob += pulp.lpSum([flex_vars[i] for i in df.index if str(df.loc[i, pos_col]).strip().upper() == 'QB']) == 0

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    return prob, status, player_vars, flex_vars

# ==========================================
# 🌐 INTERACTIVE STREAMLIT APPLICATION
# ==========================================
st.set_page_config(page_title="Universal DFS ILP Engine", page_icon="📈", layout="wide")

st.title("📈 Universal Fantasy Optimization Engine")
st.markdown("An open-source Operations Research platform. Upload **any custom platform CSV matrix** to calculate the globally optimal roster strategy under rigid financial constraints.")

# Setup Layout Columns
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("⚙️ Configuration Workspace")
    
    # 1. Interactive Budget Constraint Control
    budget_slider = st.sidebar.slider("Select Salary Cap Threshold ($)", min_value=10000, max_value=100000, value=50000, step=500)
    
    # 2. File Uploading Component
    uploaded_file = st.file_uploader("Upload your custom player data dataset (CSV format)", type=["csv"])
    
    # Fallback Mechanism: Load mock data if no file is provided
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.info("📂 Custom player data matrix successfully parsed!")
    else:
        st.warning("ℹ️ Running simulation mode. Upload a custom file to replace mock data pools.")
        # FIXED: Populated the numeric values for the Market_Salary key array completely
        mock_data = {
            "Athlete_Name": ["Patrick Mahomes", "Lamar Jackson", "Saquon Barkley", "Christian McCaffrey", "Breece Hall", "Justin Jefferson", "CeeDee Lamb", "Amon-Ra St. Brown", "Travis Kelce", "Trey McBride"],
            "Roster_Pos": ["QB", "QB", "RB", "RB", "RB", "WR", "WR", "WR", "TE", "TE"],
            "Market_Salary": [7500, 8000, 8200, 9000, 7800, 9000, 8800, 8400, 6800, 6200],
            "Expected_Points": [24.2, 26.5, 19.5, 23.1, 18.2, 21.3, 20.8, 19.1, 15.2, 14.8]
        }
        raw_df = pd.DataFrame(mock_data)

    # 3. Dynamic Structural Schema Mapper
    st.write("---")
    st.markdown("#### 🧹 Map Dataset Variables")
    st.caption("Tell the optimization math solver which columns contain the required optimization data:")
    
    available_columns = list(raw_df.columns)
    
    # Automatically try to guess the index mapping to save user time
    def find_default(options, target_keywords):
        for i, opt in enumerate(options):
            if any(key in str(opt).lower() for key in target_keywords):
                return i
        return 0

    name_field = st.selectbox("Player Name Attribute:", available_columns, index=find_default(available_columns, ["name", "player", "athlete"]))
    pos_field = st.selectbox("Positional Designation Attribute:", available_columns, index=find_default(available_columns, ["pos", "role"]))
    sal_field = st.selectbox("Financial Cost / Salary Attribute:", available_columns, index=find_default(available_columns, ["sal", "cost", "price"]))
    proj_field = st.selectbox("Expected Performance Projection Attribute:", available_columns, index=find_default(available_columns, ["proj", "point", "fp", "score"]))

with col_right:
    st.subheader("📊 Ingested System Matrix")
    # Display the current data asset cleanly
    st.dataframe(raw_df, use_container_width=True, height=300)
    
    # Action Trigger Button
    st.write("---")
    if st.button("🚀 Compute Mathematically Optimal Strategy Portfolio", type="primary", use_container_width=True):
        
        # Coerce types to float/int to protect linear programmer logic from text strings
        raw_df[sal_field] = pd.to_numeric(raw_df[sal_field], errors='coerce')
        raw_df[proj_field] = pd.to_numeric(raw_df[proj_field], errors='coerce')
        raw_df = raw_df.dropna(subset=[sal_field, proj_field])
        
        # Fire solver
        prob, status, player_vars, flex_vars = optimize_lineup(
            raw_df, budget_slider, name_field, pos_field, sal_field, proj_field
        )
        
        if pulp.LpStatus[status] == "Infeasible":
            st.error(f"❌ Structural Infeasibility Detected: The matrix conditions and positional requirements cannot be satisfied under a budget constraint of {budget_slider}. Adjust parameters or increase the salary threshold.")
        else:
            st.success("🏆 Global Maximum Allocation Matrix Formulated Successfully!")
            
            # Map solver output vectors back into structural rows
            selected_records = []
            for i in raw_df.index:
                if player_vars[i].varValue == 1:
                    row = raw_df.loc[i].copy()
                    row['Assigned Roster Slot'] = row[pos_field]
                    selected_records.append(row)
                if flex_vars[i].varValue == 1:
                    row = raw_df.loc[i].copy()
                    row['Assigned Roster Slot'] = f"FLEX ({row[pos_field]})"
                    selected_records.append(row)
                    
            output_df = pd.DataFrame(selected_records)
            
            # Render Optimized Lineup Presentation Screen
            st.subheader("🥇 Engineered Resource Allocation Matrix")
            st.dataframe(output_df[['Assigned Roster Slot', name_field, sal_field, proj_field]], use_container_width=True)
            
            # Real-Time Operational Performance Metrics
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Total Budget Capital Disbursed", f"${int(output_df[sal_field].sum()):,}")
            m_col2.metric("Optimized Expected Return Yield", f"{output_df[proj_field].sum():.2f} pts")
            m_col3.metric("Remaining Safety Surplus", f"${int(budget_slider - output_df[sal_field].sum()):,}")

st.sidebar.write("---")
st.sidebar.markdown(f'![Visitor count](https://herokuapp.com)')
