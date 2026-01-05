import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import glob

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATTERN = os.path.join(BASE_DIR, "*.csv")
MODEL_PATH = os.path.join(BASE_DIR, "admission_model.pkl")

def get_admission_files():
    csv_files = glob.glob(CSV_PATTERN)
    admission_files = [
        f for f in csv_files 
        if "placement" not in os.path.basename(f).lower() 
        and ("apeapcet" in os.path.basename(f).lower() or "approval" in os.path.basename(f).lower())
    ]
    return admission_files

def parse_data():
    files = get_admission_files()
    if not files:
        print("No Admission CSV files found!")
        return None

    csv_path = files[0]
    print(f"Parsing Admission Data from: {csv_path}")
    
    data = []
    
    with open(csv_path, "r", encoding="utf-8-sig", errors="ignore") as f:
        for line in f:
            clean_line = line.strip().strip('"')
            tokens = clean_line.split()
            if len(tokens) < 8: continue
                
            try:
                # 1. Branch
                branch = tokens[-1]
                if not (len(branch) >= 2 and len(branch) <= 5 and branch.isalpha() and branch.isupper()):
                    continue
                
                # 2. Rank
                rank = -1
                for i in range(2, 6): 
                    if tokens[i].isdigit() and int(tokens[i]) > 1 and int(tokens[i]) < 200000:
                         rank = int(tokens[i])
                         break
                if rank == -1: continue

                # 3. Gender
                gender = None
                for token in tokens:
                    if token in ['M', 'F']:
                        gender = token
                        break
                if not gender: continue

                # 4. Category
                category = None
                cats = ['OC', 'BC_A', 'BC_B', 'BC_C', 'BC_D', 'BC_E', 'SC', 'ST']
                for token in tokens:
                    for c in cats:
                        if token.startswith(c):
                            category = c
                            break
                    if category: break
                if not category: category = 'OC'
                
                if category == 'OC' and "EWS" in clean_line.upper():
                    category = 'OC_EWS'

                upper_line = clean_line.upper()
                if any(x in upper_line for x in ["NCC", "CAP", "PH", "SP"]):
                    continue

                data.append({
                    'Rank': rank,
                    'Branch': branch,
                    'Gender': gender,
                    'Category': category,
                    'Eligible': 1
                })

            except Exception:
                continue
                
    return pd.DataFrame(data)

def generate_synthetic_data(df):
    print("Generating synthetic samples with Outlier Filtering...")
    final_data = []
    
    # Ensure all key categories are present to avoid OHE errors later?
    # OneHotEncoder handle_unknown='ignore' will help, but better to have data.
    
    groups = df.groupby(['Branch', 'Gender', 'Category'])
    
    for (branch, gender, category), group in groups:
        ranks = sorted(group['Rank'])
        
        # 1. POSITIVE SAMPLES CLEANUP (85th Percentile)
        percentile_idx = int(len(ranks) * 0.85)
        cutoff_rank = ranks[percentile_idx] if ranks else 0
        
        valid_positives = group[group['Rank'] <= cutoff_rank]
        for _, row in valid_positives.iterrows():
            final_data.append(row.to_dict())
            
        # 2. NEGATIVE SAMPLES
        # Generate enough negatives to balance
        n_neg = len(valid_positives) + 5
        
        start_rank = cutoff_rank + 100
        end_rank = min(start_rank + 40000, 250000) 
        
        if start_rank >= end_rank:
             start_rank = cutoff_rank + 50
             end_rank = start_rank + 10000
             
        fake_ranks = np.random.randint(start_rank, end_rank, size=n_neg)
        
        for r in fake_ranks:
            final_data.append({
                'Rank': r,
                'Branch': branch,
                'Gender': gender,
                'Category': category,
                'Eligible': 0
            })
            
    return pd.DataFrame(final_data)

def train_and_save():
    raw_df = parse_data()
    if raw_df is None or raw_df.empty:
        print("Failed to load data.")
        return

    print(f"Loaded {len(raw_df)} raw samples.")
    full_df = generate_synthetic_data(raw_df)
    print(f"Refined dataset size: {len(full_df)} samples.")
    
    # USE ONE-HOT ENCODING via ColumnTransformer
    # This ensures "CSE" is not just "1", but [1, 0, 0...] (Linear Independence)
    
    # We define the columns to transform
    # handle_unknown='ignore' ensures if a new branch/cat appears, it doesn't crash (just gets 0s)
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False) 
    
    column_trans = make_column_transformer(
        (ohe, ['Branch', 'Gender', 'Category']),
        remainder='passthrough' # Keep Rank (which is the only numeric)
    )
    
    # Re-order DF to match [Rank, Branch, Gender, Category] passed ???
    # No, ColumnTransformer selects by name.
    
    X = full_df[['Rank', 'Branch', 'Gender', 'Category']]
    y = full_df['Eligible']
    
    # Pipeline: Transform -> LogisticRegression
    # Increased max_iter for convergence
    model = make_pipeline(column_trans, LogisticRegression(max_iter=1000))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Logistic Regression with One-Hot features...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc:.4f}")
    
    # Sanity Check
    try:
        # We pass a DataFrame or dict-like object to predict because of ColumnTransformer
        # Test 1: Good Rank (5000, CSE, OC, M) -- Should be HIGH
        test_df = pd.DataFrame([{
            'Rank': 5000,
            'Branch': 'CSE',
            'Gender': 'M',
            'Category': 'OC'
        }])
        prob = model.predict_proba(test_df)[0][1]
        print(f"Sanity (Rank 5000, CSE, M, OC): Eligible Prob = {prob:.4f}")
        
        # Test 2: Bad Rank (40000, CSE, OC, M) -- Should be LOW
        test_df2 = pd.DataFrame([{
            'Rank': 40000,
            'Branch': 'CSE',
            'Gender': 'M',
            'Category': 'OC'
        }])
        prob_bad = model.predict_proba(test_df2)[0][1]
        print(f"Sanity (Rank 40000, CSE, M, OC): Eligible Prob = {prob_bad:.4f}")

        # Test 3: Compare Category (OC vs BC_E) at same rank
        # BC_E should logically have higher probability for same rank IF cutoff is higher.
        test_df3 = pd.DataFrame([{
            'Rank': 40000,
            'Branch': 'CSE',
            'Gender': 'M',
            'Category': 'BC_E'
        }])
        prob_bce = model.predict_proba(test_df3)[0][1]
        print(f"Sanity (Rank 40000, CSE, M, BC_E): Eligible Prob = {prob_bce:.4f}")

    except Exception as e:
        print(f"Sanity check failed: {e}")

    # Save the entire pipeline (includes encoding)
    joblib.dump(model, MODEL_PATH)
    print(f"Model pipeline saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_and_save()
