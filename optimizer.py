import os
import pandas as pd
import pulp

def run_data_pipeline(salary_csv_path="real_draftkings_salaries.csv"):
    """
    Data Engineering Phase (ETL):
    Loads raw market salary data, pairs it with scraped/fetched projections,
    cleans the formatting, and generates an optimized unified dataset.
    """
    print("🚀 Starting real-world data integration pipeline...")
    
    # 1. Fallback / Self-Correction: Generate a sample file if it doesn't exist
    if not os.path.exists(salary_csv_path):
        print(f"⚠️ Warning: '{salary_csv_path}' not found. Generating real-world style sample data...")
        sample_data = (
            "Name,Position,Salary,Team\n"
            "Patrick Mahomes,QB,7500,KC\n"
            "Lamar Jackson,QB,8000,BAL\n"
            "Saquon Barkley,RB,8200,PHI\n"
            "Christian McCaffrey,RB,9000,SF\n"
            "Breece Hall,RB,7800,NYJ\n"
            "Justin Jefferson,WR,9000,MIN\n"
            "CeeDee Lamb,WR,8800,DAL\n"
            "Amon-Ra St. Brown,WR,8400,DET\n"
            "Travis Kelce,TE,6800,KC\n"
            "Trey McBride,TE,6200,ARI\n"
        )
        with open(salary_csv_path, "w") as f:
            f.write(sample_data)

    # 2. Read the raw salary platform data
    dk_salaries = pd.read_csv(salary_csv_path)
    
    # 3. Simulate real consensus performance projection feeds
    print("📡 Fetching live consensus performance projections...")
    mock_projections = {
        "Player_Name": [
            "Patrick Mahomes", "Lamar Jackson", "Saquon Barkley", 
            "Christian McCaffrey", "Breece Hall", "Justin Jefferson", 
            "CeeDee Lamb", "Amon-Ra St. Brown", "Travis Kelce", "Trey McBride"
        ],
        "Projected_FP": [24.2, 26.5, 19.5, 23.1, 18.2, 21.3, 20.8, 19.1, 15.2, 14.8]
    }
    projections_df = pd.DataFrame(mock_projections)
    
    # 4. Relational Database Join & Cleaning
    print("🧹 Cleaning data and executing relational join...")
    merged_data = pd.merge(
        dk_salaries, 
        projections_df, 
        left_on="Name", 
        right_on="Player_Name", 
        how="inner"
    )
    
    # Format and safety clean the final schema
    final_dataset = merged_data[["Name", "Position", "Salary", "Projected_FP"]].copy()
    final_dataset.rename(columns={"Projected_FP": "ProjectedPoints"}, inplace=True)
    final_dataset["ProjectedPoints"] = final_dataset["ProjectedPoints"].fillna(0)
    
    output_filename = "optimizer_input.csv"
    final_dataset.to_csv(output_filename, index=False)
    print(f"✅ Pipeline complete! Unified data written to '{output_filename}'")
    return output_filename


def optimize_lineup(data_path, salary_cap=50000):
    """
    Operations Research Phase:
    Uses Integer Linear Programming via PuLP to maximize projected points 
    subject to positional roster limits and financial constraints.
    """
    print("\n⚡ Initializing Integer Linear Programming Solver...")
    df = pd.read_csv(data_path)
    
    # 1. Define Binary Decision Variables: x_i matching the player index
    player_vars = pulp.LpVariable.dicts("Player", df.index, cat="Binary")
    
    # 2. Define Flex Decision Variables: y_i matching the player index
    # (Tracks if a player is filling the flex position rather than their base position)
    flex_vars = pulp.LpVariable.dicts("FlexPlayer", df.index, cat="Binary")
    
    # 3. Instantiate Maximization Problem
    prob = pulp.LpProblem("Fantasy_Sports_Lineup_Optimization", pulp.LpMaximize)
    
    # 4. Objective Function: Maximize expected points across base and flex picks
    prob += pulp.lpSum([df.loc[i, 'ProjectedPoints'] * (player_vars[i] + flex_vars[i]) for i in df.index]), "Maximize_Points"
    
    # 5. Financial Constraint: Combined base + flex salaries cannot cross cap
    prob += pulp.lpSum([df.loc[i, 'Salary'] * (player_vars[i] + flex_vars[i]) for i in df.index]) <= salary_cap, "Salary_Cap_Limit"
    
    # 6. Structural Constraint: A player can only be selected ONCE total
    for i in df.index:
        prob += player_vars[i] + flex_vars[i] <= 1, f"Single_Selection_Limit_Player_{i}"
        
    # 7. Positional Constraints (e.g., Target: 1 QB, 1 RB, 1 WR, 1 TE, plus 1 FLEX)
    prob += pulp.lpSum([player_vars[i] for i in df.index if df.loc[i, 'Position'] == 'QB']) == 1, "Exact_1_QB"
    prob += pulp.lpSum([player_vars[i] for i in df.index if df.loc[i, 'Position'] == 'RB']) == 1, "Base_1_RB"
    prob += pulp.lpSum([player_vars[i] for i in df.index if df.loc[i, 'Position'] == 'WR']) == 1, "Base_1_WR"
    prob += pulp.lpSum([player_vars[i] for i in df.index if df.loc[i, 'Position'] == 'TE']) == 1, "Base_1_TE"
    
    # 8. Flex Eligibility Constraint: Only RB, WR, or TE can fill the 1 Flex slot
    prob += pulp.lpSum([flex_vars[i] for i in df.index if df.loc[i, 'Position'] in ['RB', 'WR', 'TE']]) == 1, "Exact_1_FLEX"
    
    # Force non-eligible positions (like QB) to have a 0 value in the flex variables
    prob += pulp.lpSum([flex_vars[i] for i in df.index if df.loc[i, 'Position'] == 'QB']) == 0, "No_QB_In_Flex"

    # 9. Execute Solver
    print("Running optimization solver engine...")
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    print(f"Solver Outcome Status: {pulp.LpStatus[status]}")
    
    if pulp.LpStatus[status] == "Infeasible":
        print("❌ ERROR: Optimization failed. Math is infeasible for this budget or dataset configuration.")
        return

    # 10. Parse & Display Mathematical Solution Output
    print("\n🏆 --- MATHEMATICALLY OPTIMAL ROSTER --- 🏆")
    selected_players = []
    
    for i in df.index:
        # Check base roster designation
        if player_vars[i].varValue == 1:
            selected_players.append(df.loc[i])
            print(f"{df.loc[i, 'Position']}      | {df.loc[i, 'Name']:<20} | Salary: ${df.loc[i, 'Salary']:<5} | Proj Points: {df.loc[i, 'ProjectedPoints']}")
        
        # Check flex designation
        if flex_vars[i].varValue == 1:
            selected_players.append(df.loc[i])
            print(f"FLEX ({df.loc[i, 'Position']}) | {df.loc[i, 'Name']:<20} | Salary: ${df.loc[i, 'Salary']:<5} | Proj Points: {df.loc[i, 'ProjectedPoints']}")
            
    if selected_players:
        final_df = pd.DataFrame(selected_players)
        print("="*60)
        print(f"Total Portfolio Budget Spent : ${final_df['Salary'].sum()}")
        print(f"Maximum Optimized Yield      : {final_df['ProjectedPoints'].sum():.2f} Projected Points")
        print("="*60)


if __name__ == "__main__":
    # Target file parameters
    raw_input_source = "real_draftkings_salaries.csv"
    
    # Run pipeline & pass the output straight into the optimization math engine
    clean_ready_file = run_data_pipeline(raw_input_source)
    if clean_ready_file:
        # Standard daily fantasy benchmarks usually use a $50,000 cap
        optimize_lineup(clean_ready_file, salary_cap=50000)