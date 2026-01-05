import joblib
import pandas as pd
import os
import re

# Mocking the pipeline loading to test logic first, or loading real model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ADMISSION_MODEL_PATH = os.path.join(BASE_DIR, "admission_model.pkl")

print(f"Loading model from {ADMISSION_MODEL_PATH}...")
try:
    admission_model = joblib.load(ADMISSION_MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {e}")
    admission_model = None

def predict_admission_test(message):
    if not admission_model:
        return "Admission prediction model is unavailable."

    msg = message.lower()
    print(f"\nProcessing: '{message}'")
    
    try:
        # Rank
        match_rank = re.search(r'rank\s*[:=]?\s*(\d+)', msg)
        rank = int(match_rank.group(1)) if match_rank else 0
        if rank == 0:
            nums = [int(s) for s in msg.split() if s.isdigit()]
            for n in nums:
                if n > 1000: rank = n; break
        
        print(f"Detected Rank: {rank}")
        
        if rank == 0:
            return "Please provide your Rank."

        # Gender
        gender = "M" if "male" in msg or "boy" in msg else ("F" if "female" in msg or "girl" in msg else None)
        print(f"Detected Gender: {gender}")

        # Category
        category = "OC"
        cats = ['oc_ews', 'bc_a', 'bc_b', 'bc_c', 'bc_d', 'bc_e', 'oc', 'sc', 'st']
        for c in cats:
            if c in msg.replace('-', '_'):
                category = c.upper()
                break
        if "ews" in msg and category == "OC": category = "OC_EWS"
        print(f"Detected Category: {category}")

        # Branch
        branch = "CSE" 
        branch_map = {
            "cse": "CSE", "computer": "CSE",
            "aiml": "CAI", "ai": "CAI", 
            "ece": "ECE", "electronics": "ECE",
            "eee": "EEE", "electrical": "EEE",
            "mech": "MEC", "mechanical": "MEC",
            "civil": "CIV", 
            "it": "INF", "inf": "INF",
            "ds": "CSD", "data science": "CSD",
            "csm": "CSM"
        }
        found_branch = None
        for k, v in branch_map.items():
            if k in msg:
                found_branch = v
                break
        print(f"Detected Branch: {found_branch}")
        
        if not found_branch:
             return "Please specify the branch."
             
        if not gender:
             return "Please specify your gender."

        # Predict
        input_df = pd.DataFrame([{
            'Rank': rank,
            'Branch': found_branch,
            'Gender': gender,
            'Category': category
        }])
        
        print(f"Input DF:\n{input_df}")
        
        prob = admission_model.predict_proba(input_df)[0][1]
        percent = prob * 100
        
        status = "High Chance" if percent > 70 else ("Moderate Chance" if percent > 40 else "Low Chance")
        
        return f"Prediction: {percent:.1f}% ({status})"

    except Exception as e:
        return f"Error: {e}"

# Test cases from user
queries = [
    "rank 50000 male BC_A cse"
]

for q in queries:
    print(predict_admission_test(q))
