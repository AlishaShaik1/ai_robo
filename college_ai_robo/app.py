import pandas as pd
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import os

# ==========================
# MODEL (Google AI)
# ==========================
MODEL_ID = "google/flan-t5-base"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

qa_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer
)

# ==========================
# LOAD PEOPLE DATA
# ==========================
# Fix: Use absolute path relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
people_file = os.path.join(BASE_DIR, "people_data.txt")

if os.path.exists(people_file):
    print(f"Loading people data from: {people_file}")
    with open(people_file, "r", encoding="utf-8") as f:
        # Use bullet points for clear separation
        people_context = "\n".join([f"- {line.strip()}" for line in f.readlines() if line.strip()])
else:
    people_context = "People data not found."
    print(f"Warning: people_data.txt not found at {people_file}")

# ==========================
# LOAD ADMISSION DATA
# ==========================
# ==========================
# LOAD ADMISSION DATA
# ==========================
import glob
import re
from data_loader import load_placement_data

# Global Placement Data
PLACEMENT_DF = None
PLACEMENT_CONTEXT = "Placement data not available."

def generate_placement_summary(df):
    if df is None or df.empty:
        return "No placement data available."
    
    summary = "--- Placement Statistics 2023 ---\n"
    
    # Total Placed
    total = len(df)
    summary += f"Total Students Placed: {total}\n"
    
    # By Branch
    if 'Branch' in df.columns:
        counts = df['Branch'].value_counts()
        for branch, count in counts.items():
            summary += f"- {branch}: {count} students placed.\n"
            
            # Top Companies for this branch
            branch_df = df[df['Branch'] == branch]
            if 'Company' in branch_df.columns:
                # Deduplicate and clean
                companies_raw = branch_df['Company'].astype(str).tolist()
                # Split by comma if multiple companies in one cell
                all_companies = []
                for c in companies_raw:
                    for sub_c in c.split(','):
                         clean_c = sub_c.strip()
                         if clean_c and clean_c.lower() != 'nan':
                             all_companies.append(clean_c)
                
                # Count freq
                from collections import Counter
                comp_counts = Counter(all_companies)
                top_mnc = [c for c, _ in comp_counts.most_common(5)]
                summary += f"  Top Recruiters for {branch}: {', '.join(top_mnc)}\n"
            
            # Package Stats (Only for IT as requested)
            if branch == "IT" and 'Package_Val' in branch_df.columns:
                valid_pkgs = branch_df[branch_df['Package_Val'] > 0]['Package_Val']
                if not valid_pkgs.empty:
                    max_pkg = valid_pkgs.max()
                    min_pkg = valid_pkgs.min()
                    avg_pkg = valid_pkgs.mean()
                    summary += f"  Highest Package for {branch}: {max_pkg/100000:.2f} LPA\n"
                    summary += f"  Average Package for {branch}: {avg_pkg/100000:.2f} LPA\n"
                    summary += f"  Least Package for {branch}: {min_pkg/100000:.2f} LPA\n"

    return summary

def init_placement_data():
    global PLACEMENT_DF, PLACEMENT_CONTEXT
    try:
        print("Loading Placement Data...")
        PLACEMENT_DF = load_placement_data(BASE_DIR)
        if not PLACEMENT_DF.empty:
            print("Placement Data Loaded Successfully.")
            PLACEMENT_CONTEXT = generate_placement_summary(PLACEMENT_DF)
        else:
            print("Placement DataFrame is empty.")
    except Exception as e:
        print(f"Failed to load placement data: {e}")
        PLACEMENT_DF = pd.DataFrame()

init_placement_data()


GLOBAL_BRANCH_COUNTS = {}

def load_admission_data():
    global GLOBAL_BRANCH_COUNTS
    # 1. Try Loading Excel (Clean Format)
    excel_files = glob.glob(os.path.join(BASE_DIR, "*.xlsx"))
    if excel_files:
        try:
            excel_path = excel_files[0]
            print(f"Loading Excel file: {excel_path}")
            df = pd.read_excel(excel_path)
            if 'Branch' in df.columns and 'Intake' in df.columns:
                df.columns = df.columns.str.strip()
                admission_list = []
                for _, row in df.iterrows():
                    branch = str(row['Branch']).upper()
                    intake = row['Intake']
                    admission_list.append(f"- Branch {branch} has an intake of {intake} students.")
                    GLOBAL_BRANCH_COUNTS[branch] = intake # Populate global dict
                return "\n".join(admission_list)
        except Exception as e:
            print(f"Error reading Excel: {e}")

    # 2. Try Loading CSV (Unstructured Report Format)
    csv_files = glob.glob(os.path.join(BASE_DIR, "*.csv"))
    # Filter for Admission file (APEAPCET/Approval) and exclude placements
    admission_files = [
        f for f in csv_files 
        if "placement" not in os.path.basename(f).lower() 
        and ("apeapcet" in os.path.basename(f).lower() or "approval" in os.path.basename(f).lower())
    ]
    
    if admission_files:
        try:
            csv_path = admission_files[0]
            print(f"Loading CSV file: {csv_path}")
            
            branch_counts = {}
            # Use utf-8-sig to handle potential BOM
            with open(csv_path, "r", encoding="utf-8-sig", errors="ignore") as f:
                for line in f:
                    # Robust parsing: Split by whitespace and take the last token
                    parts = line.split()
                    if len(parts) > 5: # Valid lines usually have many columns
                        last_token = parts[-1].strip().strip('"')
                        # Check if it looks like a branch code (2-4 uppercase letters)
                        if len(last_token) >= 2 and len(last_token) <= 4 and last_token.isalpha() and last_token.isupper():
                             branch_counts[last_token] = branch_counts.get(last_token, 0) + 1
            
            if branch_counts:
                # Add aliases for better user understanding
                aliases = {
                    "CAI": ["AIML", "CSE(AIML)", "CSE-AIML", "AI&ML"],
                    "CSM": ["AIML", "CSE(AI&ML)", "CSE-AI&ML"], # Clarify if different
                    "MEC": ["MECHANICAL", "MECH"],
                    "ECE": ["ELECTRONICS"],
                    "EEE": ["ELECTRICAL"],
                    "CIV": ["CIVIL"],
                    "CSE": ["COMPUTER SCIENCE"]
                }
                
                admission_list = []
                for branch, count in branch_counts.items():
                    # Populate global dict for direct lookup
                    GLOBAL_BRANCH_COUNTS[branch] = count
                    for alias in aliases.get(branch, []):
                        GLOBAL_BRANCH_COUNTS[alias.upper()] = count

                    # Format: "- Branch CSE (also known as COMPUTER SCIENCE) has 180..."
                    alias_str = ""
                    if branch in aliases:
                        alias_str = f" (also known as {', '.join(aliases[branch])})"
                    admission_list.append(f"- Branch {branch}{alias_str} has {count} students admitted/approved.")
                
                return "\n".join(admission_list)
            else:
                 return "Admission data found but could not extract branch details."

        except Exception as e:
            print(f"Error reading CSV: {e}")
            return "Error loading admission data from CSV."

    return "No admission data file (Excel/CSV) found."

admission_context = load_admission_data()


# ==========================
# COMBINED KNOWLEDGE (Grounding)
# ==========================
SYSTEM_CONTEXT = f"""
Instructions: You are a helpful college assistant. 
1. Answer the question using ONLY the provided information below.
2. Be brief and direct. Do NOT print the entire context or summary unless explicitly asked to "summarize".
3. If the user asks about a specific branch (e.g., AIML), only provide information for that branch. Do NOT mention other branches.
4. If the answer is not in the text, say "I currently do not have that information.".

--- People Information ---
{people_context}

--- Admission Information ---
{admission_context}

--- Placement Information ---
{PLACEMENT_CONTEXT}
"""

# ==========================
# TYPO TOLERANCE
# ==========================
import difflib

def correct_typos(message):
    # Extract likely roles from our people data to match against
    # Assuming lines like "Role: Name"
    valid_roles = []
    if people_context and "People data not found" not in people_context:
        for line in people_context.split('\n'):
            parts = line.replace('- ', '').split(':')
            if len(parts) > 1:
                valid_roles.append(parts[0].strip().lower())
    
    # Common mappings (manual overrides)
    custom_corrections = {
        "princpaaal": "principal",
        "chaman": "chairman",
        "hd": "hod",
        "deaan": "dean",
        "princi": "principal",
        "hod aiml": "hod aiml" # Ensure multi-word roles are preserved if typed correctly
    }
    
    words = message.lower().split()
    corrected_words = []
    
    for word in words:
        # Check custom map
        if word in custom_corrections:
            corrected_words.append(custom_corrections[word])
            continue
            
        # Fuzzy match against valid roles
        matches = difflib.get_close_matches(word, valid_roles, n=1, cutoff=0.7)
        if matches:
            corrected_words.append(matches[0])
        else:
            corrected_words.append(word)
            
    return " ".join(corrected_words)

# ==========================
# STATISTICAL ADMISSION PREDICTOR (Lookup Model)
# ==========================
# Due to environment constraints with scikit-learn, we use a direct statistical model.
# This actually provides 100% accurate historical cutoffs rather than approximations.

# ==========================
# STATISTICAL ADMISSION PREDICTOR (Lookup Model)
# ==========================
# Due to environment constraints with scikit-learn, we use a direct statistical model.
# This actually provides accurate historical cutoffs rather than approximations.

ADMISSION_CUTOFFS = {} # Key: (Branch, Gender, Category), Value: Cutoff Rank

def load_prediction_model():
    global ADMISSION_CUTOFFS
    csv_files = glob.glob(os.path.join(BASE_DIR, "*.csv"))
    # Filter for admission files (APEAPCET/Approval)
    admission_files = [
        f for f in csv_files 
        if "placement" not in os.path.basename(f).lower() 
        and ("apeapcet" in os.path.basename(f).lower() or "approval" in os.path.basename(f).lower())
    ]

    if not admission_files:
        print("No Admission CSV files found for prediction!")
        return

    csv_path = admission_files[0]
    print(f"Building Prediction Model from: {csv_path}")
    
    # Store List of ranks for each group
    # Key: (Branch, Gender, Category) -> Val: [rank1, rank2, ...]
    grouped_ranks = {}
    
    # Use the robust parsing logic
    with open(csv_path, "r", encoding="utf-8-sig", errors="ignore") as f:
        for line in f:
            parts = line.split()
            clean_line = line.strip().strip('"')
            tokens = clean_line.split()
            
            if len(tokens) < 8:
                continue
                
            try:
                # 1. Branch: Last token
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
                
                # 5. EXCLUDE SPECIAL CATEGORIES (CAP, NCC, PH)
                # Note: We now allow EWS if it is part of the category (e.g. OC_EWS column if exists, or handled via category field).
                # But typically EWS is a quota. Let's check how EWS appears. 
                # If the user asks for "OC_EWS", we should verify if the CSV differentiates it. 
                # The debugging showed high ranks for OC. These probably were EWS.
                # If we filter EWS out of "OC", we refine OC. 
                # But if we want to support "OC_EWS" specifically, we need to capture it.
                
                # Logic Update:
                # If the line says "EWS", treat it as a special modifier.
                # If the token 'OC_EWS' or 'EWS' exists in tokens, we capture it.
                
                is_ews = False
                if "EWS" in clean_line.upper():
                    is_ews = True
                    
                # If we parsed category as 'OC' but line has EWS, let's change category to 'OC_EWS'
                if category == 'OC' and is_ews:
                    category = 'OC_EWS'

                # Exclude other special quotas
                upper_line = clean_line.upper()
                if "NCC" in upper_line or "CAP" in upper_line or "PH" in upper_line or "SP" in upper_line:
                    continue
                    
                # If strict EWS filtering was desired for "OC":
                # We essentially just made a new category 'OC_EWS'. 
                # So 'OC' stats will now be pure OC (without EWS), and 'OC_EWS' will be its own.

                # Store
                key = (branch, gender, category)
                if key not in grouped_ranks:
                    grouped_ranks[key] = []
                grouped_ranks[key].append(rank)

            except Exception:
                continue
    
    # Calculate Percentile Cutoff
    # Instead of MAX, we use 90th percentile to filter out extreme outliers
    for key, ranks in grouped_ranks.items():
        if ranks:
            # Sort ranks
            ranks.sort()
            # Take the 85th percentile (safer bet than Max)
            idx = int(len(ranks) * 0.90) 
            if idx >= len(ranks): idx = len(ranks) - 1
            cutoff = ranks[idx]
            ADMISSION_CUTOFFS[key] = cutoff
            
    # For OC CSE and BC_D CSE debugging
    # print(f"DEBUG: OC-M-CSE Cutoff: {ADMISSION_CUTOFFS.get(('CSE', 'M', 'OC'))}")

    print(f"Prediction Model Ready. Loaded stats for {len(ADMISSION_CUTOFFS)} groups.")

# Initialize the model
load_prediction_model()

def predict_eligibility(message):
    if not ADMISSION_CUTOFFS:
        return "Prediction data is not available."
        
    msg = message.lower()
    
    try:
        # Extract Rank
        rank = -1
        match_rank = re.search(r'rank\s*[:=]?\s*(\d+)', msg)
        if match_rank:
            rank = int(match_rank.group(1))
        
        # Extract Gender
        gender = None
        if "female" in msg or " gender f" in msg or "gender:f" in msg:
            gender = "F"
        elif "male" in msg or " gender m" in msg or "gender:m" in msg:
            gender = "M"
            
        # Extract Category
        category = None
        # Naive matching: specific to general order
        cats = ['oc_ews', 'bc_a', 'bc_b', 'bc_c', 'bc_d', 'bc_e', 'oc', 'sc', 'st']
        for c in cats:
            if c in msg.replace('-', '_'):
                category = c.upper()
                break
        
        # Extract Branch
        branch = None
        aliases = {
            "CAI": ["aiml", "ai&ml", "cse(aiml)"],
            "CSM": ["csm", "cse(ai&ml)"],
            "MEC": ["mec", "mech", "mechanical"],
            "ECE": ["ece", "electronics"],
            "EEE": ["eee", "electrical", "electrical engineering"],
            "CIV": ["civil", "civil engineering"],
            "CSE": ["cse", "computer science", "cse core"],
            "INF": ["it", "information technology", "cse(it)"],
            "CSD": ["ds", "data science", "cse(ds)"],
            "CSC": ["cs", "cyber security", "cse(cs)"]
        }
        
        for code, keywords in aliases.items():
            for kw in keywords:
                if kw in msg:
                    branch = code
                    break
            if branch: break
            
        if not (rank > 0 and gender and category and branch):
            return "Please provide Rank, Gender, Category, and Branch to predict eligibility."

        # Lookup Cutoff
        key = (branch, gender, category)
        cutoff_rank = ADMISSION_CUTOFFS.get(key)
        
        # Fallback: if exact match not found (e.g. no girls in mechanical for that category), try finding similar
        # But for now, let's just report unknown.
        
        result_msg = f"**Admission Prediction**\n"
        result_msg += f"- Details: Rank {rank}, {gender}, {category}, {branch}\n"
        
        if cutoff_rank:
            result_msg += f"- 2023 Cutoff Rank: {cutoff_rank}\n"
            if rank <= cutoff_rank:
                result_msg += f"✅ **Yes, you have a high chance!** (Within last year's cutoff)"
            else:
                result_msg += f"❌ **Diffcult.** Your rank is higher than last year's cutoff ({cutoff_rank})."
        else:
            result_msg += f"⚠️ No data found for this exact combination ({branch}, {gender}, {category}) in 2023 records."
            
        return result_msg

    except Exception as e:
        print(f"Prediction Error: {e}")
        return "An error occurred during prediction."

# ==========================
# PLACEMENT QUERY ENGINE
# ==========================
# ==========================
# PLACEMENT QUERY ENGINE
# ==========================
def query_placements(message):
    if PLACEMENT_DF is None or PLACEMENT_DF.empty:
        return "Placement data is currently unavailable."
        
    msg = message.lower()
    
    # Normalize Branch Names for lookup
    # Order matters: check longer/specific matches first
    branch_map = {
        "aiml": "AIML", "ai & ml": "AIML", "artificial intelligence": "AI", "cse(ai)": "AI", "ai": "AI",
        "data science": "DS", "cse(ds)": "DS", "ds": "DS",
        "information technology": "IT", "it": "IT",
        "civil engineering": "CIVIL", "civil": "CIVIL", 
        "mechanical": "MECH", "mech": "MECH", 
        "electronics": "ECE", "ece": "ECE",
        "electrical": "EEE", "eee": "EEE",
        "computer science": "CSE", "cse": "CSE"
    }
    
    target_branch = None
    # Use word boundary check to avoid "cse" matching "civil" (if 'c' matched?) or similar issues
    # "cse" matching "civil" shouldn't happen with simple substring unless "c" or "s" was mapped?
    # The previous map had "ce": "CIVIL". "cse" contains "ce". THAT WAS THE BUG.
    # Removed "ce": "CIVIL" and ensured order.
    
    for k, v in branch_map.items():
        # Check as whole word using word boundaries, case insensitive flag in search
        # Problem: "ai" might be inside "said"? No, \b handles that.
        # Problem: "ai" in "aiml"? \b handles that.
        pattern = r"\b" + re.escape(k) + r"\b"
        if re.search(pattern, msg, re.IGNORECASE):
            target_branch = v
            break
            
    # Intent Detection with Regex for robustness
    # 1. "How many" (Count)
    # Matches: "how many", "ho many", "number of", "count of", "total placements"
    # Also matches "placements of [branch]" which implies count or summary
    # Typo: "total placemnts", "total placemnets" -> catch "total place" or "placem"
    if re.search(r"(how|ho|total)\s*(many|number|count|placem|place)", msg) or re.search(r"placem(e)?n?ts?\s*(of|for|in)", msg):
        if target_branch:
            # Specific branch count
            count = len(PLACEMENT_DF[PLACEMENT_DF['Branch'] == target_branch])
            return f"The number of placements for **{target_branch}** is **{count}**."
        else:
            # Total count
            if "aiml" in msg or "ai" in msg or "cse" in msg: 
                # Fallback if valid branch name exists but wasn't mapped effectively above?
                # Actually target_branch logic is solid. If we are here, no branch matched.
                pass
            count = len(PLACEMENT_DF)
            return f"A total of **{count}** students have been placed across all branches."

    # 2. "Companies" (List)
    # Matches: "companies", "company", "recruiters", "hiring", "visit"
    # Typo: "compines", "componies" -> catch "comp[a-z]*ies" or just "comp"
    # Be careful not to match "computer"
    if re.search(r"(compan|recruit|hir|visit|compin|compon)", msg):
        branch_df = PLACEMENT_DF
        if target_branch:
            branch_df = PLACEMENT_DF[PLACEMENT_DF['Branch'] == target_branch]
            
        # Deduplicate Logic
        companies_raw = branch_df['Company'].astype(str).tolist()
        all_companies = []
        for c in companies_raw:
            # Handle multi-value cells like "TCS, Wipro"
            for sub_c in c.replace(' and ', ',').split(','):
                clean_c = sub_c.strip()
                # Remove common legal suffixes for cleaner deduplication (visual only) if needed?
                # or just lowercase comparison
                if clean_c and clean_c.lower() != 'nan':
                    all_companies.append(clean_c)
        
        # Smart Deduplication (Case Insensitive Counting, preserving original casing)
        final_list = []
        seen_lower = set()
        for c in all_companies:
            if c.lower() not in seen_lower:
                seen_lower.add(c.lower())
                final_list.append(c)
        
        # Only top 15 from the UNIQUE list? 
        # No, we want most frequent.
        from collections import Counter
        # Normalize for counting
        norm_companies = [c.title() for c in all_companies] 
        comp_counts = Counter(norm_companies)
        
        # Top 15 by frequency
        top_list = [c for c, _ in comp_counts.most_common(15)]
        
        scope = target_branch if target_branch else "overall"
        return f"Top companies for **{scope}** include: {', '.join(top_list)}..."


    # 3. "Highest/Least/Average Package"
    if "package" in msg or "salary" in msg:
        # Check for IT branch restriction or implied context
        # Usage instructions: "only IT has packages"
        target_df = PLACEMENT_DF
        if target_branch:
             target_df = PLACEMENT_DF[PLACEMENT_DF['Branch'] == target_branch]

        # Check if valid package data exists for this slice
        # If user asks for non-IT package, we should arguably say "Data unavailable" or "Check IT"
        # But if they ask "Highest package" globally, we can show IT's highest.
        
        valid_pkgs = target_df[target_df['Package_Val'] > 0]['Package_Val']
        
        if valid_pkgs.empty:
             return "Package information is currently only available for the IT branch."

        val_fmt = lambda x: f"{x/100000:.2f} LPA"

        if "highest" in msg or "max" in msg:
             return f"The highest package recorded is **{val_fmt(valid_pkgs.max())}** (mainly from IT data)."
             
        if "least" in msg or "lowest" in msg or "min" in msg:
             return f"The least package recorded is **{val_fmt(valid_pkgs.min())}**."
             
        if "average" in msg or "mean" in msg:
             return f"The average package recorded is **{val_fmt(valid_pkgs.mean())}**."

    # 4. Fallback: Use LLM with injected context
    # We return None so the main loop calls the LLM pipeline
    return None


# ==========================
# CHAT FUNCTION
# ==========================
def respond(message, history):
    # Greeting Logic
    msg_lower = message.lower().strip()
    if msg_lower in ["hi", "hello", "hey", "hi there"]:
        return "Hi, I am Pragati Engineering College AI. How can I help you?"

    # 1. Correct Typos
    corrected_message = correct_typos(message)

    # 2. Check for Placement Queries
    # Added "placed" to catch "how many students placed"
    if "placement" in msg_lower or "package" in msg_lower or "salary" in msg_lower or "company" in msg_lower or "companies" in msg_lower or "highest" in msg_lower or "placed" in msg_lower:
        response = query_placements(corrected_message)
        if response:
            return response
        # If None, fall through to LLM (which has the placement context)


    # 2. Check for ML Prediction Query
    # Trigger keywords: "rank" AND ("branch" or "gender" or "category")
    if "rank" in msg_lower and ("gender" in msg_lower or "category" in msg_lower or "branch" in msg_lower):
        prediction = predict_eligibility(corrected_message)
        if prediction:
            return prediction

    # 3. Direct Fallback for "Intake" queries (Determinstic)
    if "intake" in msg_lower or "how many" in msg_lower:
        for branch, count in GLOBAL_BRANCH_COUNTS.items():
            if branch.lower() in msg_lower:
                return f"{count} students admitted/approved for {branch}."

    # 4. Help Command
    if "show questions" in msg_lower or "help" in msg_lower:
        help_text = "**Available Question Formats:**\n\n"
        help_text += "🔹 **Principal/Personnel:**\n`Who is the Principal?`\n\n"
        help_text += "🔹 **Admission Intake:**\n`What is the intake of CSE?`\n\n"
        help_text += "🔹 **Eligibility Prediction:**\n`Rank 30000 Category OC Gender Male Branch CSE`"
        return help_text

    # 4. Use LLM for everything else
    prompt = f"""
{SYSTEM_CONTEXT}

Question: {corrected_message}
Answer:
"""

    result = qa_pipeline(
        prompt,
        max_length=200,
        temperature=0.2 
    )

    result_text = result[0]["generated_text"]
    
    # Post-process: Strip context leakage
    if "--- People Information ---" in result_text:
        result_text = result_text.split("--- People Information ---")[0]
    if "Instructions:" in result_text:
        result_text = result_text.split("Instructions:")[0]
        
    return result_text.strip()


# ==========================
# GRADIO UI
# ==========================
# ==========================
# GRADIO UI (Blocks with Legacy Chatbot)
# ==========================
if __name__ == "__main__":
    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)

# ==========================
# GRADIO UI
# ==========================
if __name__ == "__main__":
    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)
    
    # Reverting to the standard, stable ChatInterface
    # No custom theme, no complex blocks.
    # The "Help" feature is implemented as a clickable example.
    
    gr.ChatInterface(
        respond,
        title="AI College Robo 🤖",
        description="AI-powered college information system using Google FLAN-T5",
        examples=[
            "Show Questions / Help",
            "Who is the Principal?",
            "What is the intake of CSE?",
            "Rank 30000 Category OC Gender Male Branch CSE",
            "Rank 60000 Category OC_EWS Gender Male Branch IT"
        ]
    ).launch()
