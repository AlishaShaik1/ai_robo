import pandas as pd
import glob
import os
import re

def clean_money_string(val):
    """
    Parses salary strings to float value in Rupees.
    Examples: 
      '3.5 LPA' -> 350000.0
      '50K PM' -> 50000 * 12 = 600000.0
      '4-6 LPA' -> 600000.0 (Taking max for highest/cutoff checks)
    """
    if pd.isna(val):
        return 0.0
    s = str(val).upper().replace(',', '').strip()
    
    # helper to parse number
    def get_max_num(text):
        matches = re.findall(r"[\d\.]+", text)
        if not matches: return 0.0
        # If range "4-6", take the last one (max) as it represents potential? 
        # Or should we take average? For "Highest Package" visualization, max is better.
        # But for "Average Package" calculation, taking max might skew it.
        # Let's take the first number (base) to be conservative, 
        # UNLESS the user asks for highest. 
        # User prompt: "make it correct and improve accuracy".
        # '4-6 LPA' usually means variable. I will take the AVG of the range for statistical correctness.
        nums = [float(m) for m in matches]
        return sum(nums) / len(nums)

    try:
        # Handle Per Month
        if 'PM' in s or 'MONTH' in s:
            num = get_max_num(s)
            if 'K' in s:
                num *= 1000
            return num * 12
            
        # Handle LPA / Lakhs
        if 'LPA' in s or 'LAKH' in s:
            num = get_max_num(s)
            return num * 100000
            
        # Handle raw numbers (assume LPA if small < 100, else raw)
        num = get_max_num(s)
        if num < 100: # E.g. 3.5
            return num * 100000
        return num
            
    except:
        return 0.0

def load_placement_data(base_dir):
    """
    Loads all placement CSV files, skipping metadata rows to find the real header.
    Normalizes columns and returns a combined DataFrame.
    """
    csv_files = glob.glob(os.path.join(base_dir, "*.csv"))
    print(f"Found {len(csv_files)} files to scan for placement data.")
    
    dfs = []
    
    # Column identification keywords
    col_keywords = {
        "Roll No": ["roll no", "rank", "ht no", "reg no"],
        "Name": ["student name", "name of the student", "full name", "name of student"],
        "Company": ["company", "name of the company", "company selected", "recruiter"],
        "Branch": ["branch"],
        "Package": ["package", "ctc", "salary"]
    }

    for f in csv_files:
        try:
            filename = os.path.basename(f).upper()
            
            # Skip non-placement files (like the admission ones)
            if "PLACEMENT" not in filename and "PLACEMENTS" not in filename:
                # Heuristic: if it doesn't say placement, check content later? 
                # For now, let's assume valid files have "placement" or we rely on content.
                # The user said "5 files of placements data". Let's check matching filenames.
                # Actually, the file list had: placement_AI.csv, placement_IT.csv etc.
                if not filename.startswith("PLACEMENT"):
                    continue

            print(f"Processing {filename}...")
            
            # 1. Read file as lines to find the header
            with open(f, "r", encoding="utf-8-sig", errors="ignore") as file:
                lines = file.readlines()
            
            header_index = -1
            header_line = ""
            
            for i, line in enumerate(lines):
                # Check if this line looks like a header
                lower_line = line.lower()
                # Must contain at least Roll No or Name AND Company
                has_student = any(k in lower_line for k in col_keywords["Roll No"] + col_keywords["Name"])
                has_company = any(k in lower_line for k in col_keywords["Company"])
                
                if has_student and has_company:
                    header_index = i
                    header_line = line
                    break
            
            if header_index == -1:
                print(f"  Skipping {filename}: Could not find a valid header row.")
                continue
                
            # 2. Read CSV starting from header_index
            # We use the detected line to infer column names, but better to let pandas read it
            try:
                # Re-read using pandas skipping rows
                df = pd.read_csv(f, skiprows=header_index, encoding="utf-8-sig")
            except:
                continue

            # 3. Normalize Columns
            renamed = {}
            for col in df.columns:
                c_lower = str(col).lower().strip()
                for standard_name, keywords in col_keywords.items():
                    for kw in keywords:
                        if kw in c_lower:
                            renamed[col] = standard_name
                            break
                    if col in renamed: break
            
            df.rename(columns=renamed, inplace=True)
            
            # 4. Standardize Branch
            # If Branch column is missing, infer from filename
            if "Branch" not in df.columns:
                inferred_branch = "UNKNOWN"
                if "_IT" in filename: inferred_branch = "IT"
                elif "_AI_ML" in filename: inferred_branch = "AIML"
                elif "_AI" in filename: inferred_branch = "AI" # Distinct from AIML?
                elif "_DS" in filename: inferred_branch = "DS"
                elif "_CIVIL" in filename: inferred_branch = "CIVIL"
                elif "MECH" in filename: inferred_branch = "MECH"
                elif "ECE" in filename: inferred_branch = "ECE"
                elif "EEE" in filename: inferred_branch = "EEE"
                elif "CSE" in filename: inferred_branch = "CSE"
                
                df["Branch"] = inferred_branch
            else:
                # Clean existing branch
                df["Branch"] = df["Branch"].astype(str).str.upper().str.strip()
                # Map complex names
                branch_map = {
                    "CSE(AI)": "AI",
                    "CSE(AIML)": "AIML",
                    "CSE(DS)": "DS",
                    "CSE(AI&ML)": "AIML",
                    "CSM": "AIML",
                    "CSD": "DS",
                    "CSC": "CYBER",
                    "CAI": "AI"
                }
                df["Branch"] = df["Branch"].replace(branch_map)

            # 5. Clean Company Name
            if "Company" in df.columns:
                df["Company"] = df["Company"].astype(str).str.strip()
                # Remove quotes or extra spaces
                df["Company"] = df["Company"].str.replace('"', '', regex=False)
            
            # 6. Clean Package if exists
            if "Package" in df.columns:
                df["Package_Val"] = df["Package"].apply(clean_money_string)
            else:
                df["Package_Val"] = 0.0

            # 7. Drop summary rows (often at bottom)
            # Check if "Roll No" or "Name" is valid
            if "Roll No" in df.columns:
                df = df[df["Roll No"].astype(str).str.len() > 5] # Valid Roll No usually long
            elif "Name" in df.columns:
                df = df[df["Name"].notna()]
            
            # Normalization Map for Data Loading
            # This ensures keys in app.py (AI, AIML, DS) match values in DF
            if 'Branch' in df.columns:
                def norm_branch(b):
                    b = str(b).upper().strip()
                    if "CSE" in b and "AI" in b and "ML" in b: return "AIML"
                    if "AI" in b and "ML" in b: return "AIML"
                    if "CSE" in b and "AI" in b: return "AI" # CSE(AI) -> AI
                    if "CSE" in b and ("DS" in b or "DATA" in b): return "DS"
                    if "CSE" in b and "IT" in b: return "IT" # CSE(IT) -> IT? Maybe just IT
                    if "INFORMATION" in b: return "IT"
                    if "CIVIL" in b: return "CIVIL"
                    if "MECH" in b: return "MECH"
                    if "ECE" in b: return "ECE"
                    if "EEE" in b: return "EEE"
                    return b
                
                df['Branch'] = df['Branch'].apply(norm_branch)
            
            dfs.append(df)

        except Exception as e:
            print(f"Error processing {f}: {e}")

    if not dfs:
        return pd.DataFrame()
        
    final_df = pd.concat(dfs, ignore_index=True)
    print(f"Total Placement Records: {len(final_df)}")
    return final_df

if __name__ == "__main__":
    df = load_placement_data(os.path.dirname(os.path.abspath(__file__)))
    print(df.head())
    if not df.empty:
        print("\nCounts by Branch:")
        print(df['Branch'].value_counts())
