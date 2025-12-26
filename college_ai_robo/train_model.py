import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import glob
import os
import re

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATTERN = os.path.join(BASE_DIR, "*.csv")
MODEL_PATH = os.path.join(BASE_DIR, "admission_model.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "encoders.pkl")

def parse_csv_data():
    csv_files = glob.glob(CSV_PATTERN)
    if not csv_files:
        print("No CSV files found!")
        return None

    csv_path = csv_files[0]
    print(f"Parsing CSV: {csv_path}")
    
    data = []
    
    # Use the robust logic we debugged earlier
    with open(csv_path, "r", encoding="utf-8-sig", errors="ignore") as f:
        for line in f:
            parts = line.split()
            # Heuristic: Valid line has Rank, Branch, Gender, Category
            # A typical line ends with Branch, but let's try to extract specific fields
            # Format usually: ... [RANK] ... [GENDER] [CATEGORY] [REGION] ... [BRANCH]
            
            # This is tricky because the CSV is unstructured text.
            # We see lines like: 
            # "1 50150010050 46140 KATCHALLA ... M BC_D AU Y CAI"
            
            # Let's clean the line and look for the pattern
            clean_line = line.strip().strip('"')
            tokens = clean_line.split()
            
            if len(tokens) < 8:
                continue
                
            try:
                # 1. Branch: Last token
                branch = tokens[-1]
                if not (len(branch) >= 2 and len(branch) <= 5 and branch.isalpha() and branch.isupper()):
                    continue
                
                # 2. Rank: Usually the 3rd token (index 2) or roughly there. 
                # It is a large integer. Let's find the first large integer after the HT NO.
                # Token 0: S.NO, Token 1: HT.NO (10 digits), Token 2: RANK (variable digits)
                rank = -1
                for i in range(2, 6): # Search first few tokens for rank
                    if tokens[i].isdigit() and int(tokens[i]) > 1 and int(tokens[i]) < 200000:
                         rank = int(tokens[i])
                         break
                
                if rank == -1:
                    continue

                # 3. Gender: 'M' or 'F'
                gender = None
                for token in tokens:
                    if token in ['M', 'F']:
                        gender = token
                        break
                if not gender:
                    continue

                # 4. Category: BC_A, BC_B, OC, etc.
                category = None
                # Known categories
                cats = ['OC', 'BC_A', 'BC_B', 'BC_C', 'BC_D', 'BC_E', 'SC', 'ST']
                for token in tokens:
                    # Often attached like BC_D_GEN or just BC_D
                    for c in cats:
                        if token.startswith(c):
                            category = c
                            break
                    if category:
                        break
                
                if not category:
                    category = 'OC' # Default fallback if not found
                
                data.append({
                    'Rank': rank,
                    'Branch': branch,
                    'Gender': gender,
                    'Category': category
                })

            except Exception as e:
                continue
                
    return pd.DataFrame(data)

def train():
    df = parse_csv_data()
    if df is None or df.empty:
        print("No data extracted. Training aborted.")
        return

    print(f"Extracted {len(df)} records.")
    print(df.head())
    
    # Feature Engineering
    # We want to predict the CUTOFF rank (Max Rank accepted) for a (Branch, Gender, Category)
    # Actually, for the request "Can I get a seat?", we want to know if UserRank <= Prediction.
    # So we should model the MAX rank for a group.
    
    # Group by Branch, Gender, Category and find the MAX rank.
    cutoff_df = df.groupby(['Branch', 'Gender', 'Category'])['Rank'].max().reset_index()
    cutoff_df.rename(columns={'Rank': 'CutoffRank'}, inplace=True)
    
    print("Training data sample (Cutoffs):")
    print(cutoff_df.head())

    # Preprocessing
    le_branch = LabelEncoder()
    le_gender = LabelEncoder()
    le_category = LabelEncoder()
    
    cutoff_df['Branch_Enc'] = le_branch.fit_transform(cutoff_df['Branch'])
    cutoff_df['Gender_Enc'] = le_gender.fit_transform(cutoff_df['Gender'])
    cutoff_df['Category_Enc'] = le_category.fit_transform(cutoff_df['Category'])
    
    X = cutoff_df[['Branch_Enc', 'Gender_Enc', 'Category_Enc']]
    y = cutoff_df['CutoffRank']
    
    # Train Model
    # Using Random Forest Regressor to capture non-linear relationships
    # High number of estimators for stability
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    print("Model trained successfully.")
    
    # Save artifacts
    artifacts = {
        'model': model,
        'le_branch': le_branch,
        'le_gender': le_gender,
        'le_category': le_category
    }
    
    joblib.dump(artifacts, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    with open("debug_log.txt", "w") as log:
        log.write("Starting training script...\n")
        try:
            train()
            log.write("Training completed.\n")
        except Exception as e:
            log.write(f"CRASHED: {e}\n")
