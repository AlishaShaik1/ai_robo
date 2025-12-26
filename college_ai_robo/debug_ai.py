from app import init_placement_data, PLACEMENT_DF, query_placements
from data_loader import load_placement_data
import os

print("--- Debugging AI Branch Issue ---")

# 1. Load Data
init_placement_data()
if PLACEMENT_DF is not None:
    print(f"Unique Branches in DF: {PLACEMENT_DF['Branch'].unique()}")
    ai_df = PLACEMENT_DF[PLACEMENT_DF['Branch'] == 'AI']
    print(f"Rows with Branch='AI': {len(ai_df)}")
    cse_ai_df = PLACEMENT_DF[PLACEMENT_DF['Branch'] == 'CSE(AI)']
    print(f"Rows with Branch='CSE(AI)': {len(cse_ai_df)}")

# 2. Test Logic directly
msg = "how many placements for ai"
print(f"\nTesting Query: '{msg}'")
response = query_placements(msg)
print(f"Response: {response}")
