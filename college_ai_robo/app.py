import pandas as pd
import gradio as gr
import joblib
import os
import re
from vector_store import CollegeKnowledgeBase
import train_intent
import train_admission
from data_loader import load_placement_data

# ==========================
# SETUP & MODEL LOADING
# ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTENT_MODEL_PATH = os.path.join(BASE_DIR, "intent_model.pkl")
ADMISSION_MODEL_PATH = os.path.join(BASE_DIR, "admission_model.pkl")

print("--- AI ROBO STARTUP ---")

# 1. Intent Model
if not os.path.exists(INTENT_MODEL_PATH):
    print("Intent model not found. Training now...")
    try:
        train_intent.train_intent_model()
    except Exception as e:
        print(f"Error training intent model: {e}")

try:
    print("Loading Intent Model...")
    intent_model = joblib.load(INTENT_MODEL_PATH)
except Exception as e:
    print(f"Failed to load intent model: {e}")
    intent_model = None

# 2. Admission Model
if not os.path.exists(ADMISSION_MODEL_PATH):
    print("Admission model not found. Training now...")
    try:
        train_admission.train_and_save()
    except Exception as e:
        print(f"Error training admission model: {e}")

try:
    print("Loading Admission Model...")
    # Loads the full pipeline (ColumnTransformer + LogReg)
    admission_model = joblib.load(ADMISSION_MODEL_PATH)
except Exception as e:
    print(f"Failed to load admission model: {e}")
    admission_model = None

# 3. Knowledge Base (FAQ)
print("Initializing Knowledge Base...")
kb = CollegeKnowledgeBase()

# 4. Placement Data (Load Data Only)
PLACEMENT_DF = load_placement_data(BASE_DIR)

# 5. People Data (Rule-based Lookup)
people_file = os.path.join(BASE_DIR, "people_data.txt")
PEOPLE_DICT = {} 
if os.path.exists(people_file):
    with open(people_file, "r", encoding="utf-8") as f:
        for line in f:
             if ":" in line:
                 parts = line.split(":", 1)
                 role = parts[0].strip().lower()
                 name = parts[1].strip()
                 # Store both ways for lookup
                 PEOPLE_DICT[role] = name
                 PEOPLE_DICT[name.lower()] = role

# ==========================
# DATA LOADING & HELPER
# ==========================
def get_intake_info(message):
    intake_data = {
        "CSE": 180,
        "AIML": 180,
        "ECE": 300,
        "EEE": 120,
        "MECH": 120,
        "CIVIL": 60,
        "IT": 60,
        "DS": 60,
        "AI": 60,
        "CYBER": 60
    }
    
    msg = message.lower()
    
    # 1. Check for specific branch using Regex Word Boundaries
    # Prevents "details" matching "ai", "place" matching "ce" etc.
    target_branch = None
    # Added "cyber" to the list
    for code in ["cse", "aiml", "ece", "eee", "mech", "civil", "it", "ds", "ai", "cyber"]:
        # \b ensures we match " ai " or "ai" at end, but not "avail" or "details"
        if re.search(r"\b" + re.escape(code) + r"\b", msg):
            target_branch = code.upper()
            break
            
    if target_branch:
        count = intake_data.get(target_branch, "Data Unavailable")
        return f"The intake (seats) for **{target_branch}** is **{count}**."
    
    # 2. Return All
    lines = ["**College Intake Details**:"]
    for branch, count in intake_data.items():
        lines.append(f"- {branch}: {count}")
    return "\n".join(lines)

def get_placement_info(message):
    if PLACEMENT_DF is None or PLACEMENT_DF.empty:
        return "Placement data unavailable."
    
    msg = message.lower()
    
    # 1. Detect Branch
    target_branch = None
    branch_map = {
        "aiml": "AIML", "ai": "AIML", "artificial intelligence": "AIML",
        "cse": "CSE", "computer": "CSE",
        "ece": "ECE", "electronics": "ECE",
        "eee": "EEE", "electrical": "EEE",
        "mech": "MECH", "mechanical": "MECH",
        "civil": "CIVIL", 
        "it": "IT", "information technology": "IT",
        "ds": "DS", "data science": "DS",
        "cyber": "CSC" # Assuming Cyber placement code is CSC or CYBER, checking DF later if needed. For now mapping to key.
    }
    
    for k, v in branch_map.items():
        if re.search(r"\b" + re.escape(k) + r"\b", msg):
            target_branch = v
            break
            
    # Filter Data
    df = PLACEMENT_DF
    if target_branch:
        if 'Branch' in df.columns:
            # Check if target_branch exists in DF, if not try aliases if needed?
            # For Cyber, it might be listed as 'CYBER' or 'CSC'.
            # Let's try flexible filtering if strict match fails?
            df_filtered = df[df['Branch'] == target_branch]
            if df_filtered.empty and target_branch == "CSC":
                 # Try fallback
                 df_filtered = df[df['Branch'] == "CYBER"]
            if not df_filtered.empty:
                df = df_filtered
            
    if df.empty:
        return f"No placement records found for {target_branch if target_branch else 'this query'}."

    # 2. Stats
    count = len(df)
    high_pkg = 0.0
    avg_pkg = 0.0
    
    if 'Package_Val' in df.columns:
        valid_pkgs = df[df['Package_Val'] > 0]['Package_Val']
        if not valid_pkgs.empty:
            high_pkg = valid_pkgs.max() / 100000
            avg_pkg = valid_pkgs.mean() / 100000
            
    # 3. Top Companies
    top_companies = []
    if 'Company' in df.columns:
        companies_raw = df['Company'].astype(str).tolist()
        all_companies = []
        for c in companies_raw:
             # Handle comma separated
             for sub_c in c.replace(',', ';').split(';'):
                 clean_c = sub_c.strip().title()
                 if clean_c and clean_c.lower() != 'nan' and len(clean_c) > 2:
                     all_companies.append(clean_c)
        
        from collections import Counter
        c_counts = Counter(all_companies)
        # Top 30 unique (User asked for "all", providing a comprehensive list)
        top_companies = [c for c, _ in c_counts.most_common(30)]
        
    # Response
    if "highest" in msg or "max" in msg:
        if high_pkg > 0:
            return f"The highest package for **{target_branch if target_branch else 'College'}** is **{high_pkg:.2f} LPA**."
        else:
            return f"Highest package info is not available for **{target_branch if target_branch else 'this query'}**."
    
    if "average" in msg:
        if avg_pkg > 0:
            return f"The average package for **{target_branch if target_branch else 'College'}** is **{avg_pkg:.2f} LPA**."
        else:
            return f"Average package info is not available for **{target_branch if target_branch else 'this query'}**."
        
    if "companies" in msg or "recruiters" in msg or "company" in msg:
        return f"**Companies for {target_branch if target_branch else 'College'}**:\n{', '.join(top_companies)}"
        
    if "count" in msg or "how many" in msg:
        return f"Total students placed: **{count}**."

    # Default Summary
    summary = f"**{target_branch if target_branch else 'Overall'} Placement Stats**:\n"
    summary += f"- Total Placed: {count}\n"
    
    if high_pkg > 0:
        summary += f"- Highest Package: {high_pkg:.2f} LPA\n"
    if avg_pkg > 0:
        summary += f"- Average Package: {avg_pkg:.2f} LPA\n"
        
    summary += f"- Top Recruiters: {', '.join(top_companies[:10])} and many more."
    return summary

def get_people_info(message):
    msg = message.lower()
    
    # Creator Info
    if "creator" in msg or "who created" in msg or "made you" in msg:
        return ("I am created by the students of AIML in 2026 on the occasion of Strides 2026 by "
                "Tharak Ram, Alisha, Rohit, Ashis, Vijay, Mahesh in the guidance of "
                "Dr. Radha Krishna Sir, V. Ananthalaksmi Mam, Janardhan Rao Sir.")

    # 1. Direct Role Lookup (Key Match)
    clean_msg = msg.replace(" of ", " ").replace(" the ", " ").replace(" is ", " ")
    
    found_role = None
    best_len = 0
    
    for role, name in PEOPLE_DICT.items():
        if role in clean_msg:
             if len(role) > best_len:
                 best_len = len(role)
                 found_role = (role, name)
                 
    if found_role:
        return f"The {found_role[0].upper()} is **{found_role[1]}**."

    # 2. Reverse Lookup (Name Match)
    ignore_TITLES = ["dr", "dr.", "mr", "mr.", "mrs", "mrs.", "sir", "mam", "prof", "prof."]
    
    for role, name in PEOPLE_DICT.items():
        name_lower = name.lower()
        name_parts = name_lower.replace('.', ' ').split()
        for part in name_parts:
            if len(part) > 2 and part not in ignore_TITLES and part in msg.split():
                 return f"**{name}** is the {role.title()}."
                
    return "I couldn't find that person in my database."

def predict_admission(message):
    if not admission_model:
        return "Admission prediction model is unavailable."

    msg = message.lower()
    
    try:
        # Rank
        match_rank = re.search(r'rank\s*[:=]?\s*(\d+)', msg)
        rank = int(match_rank.group(1)) if match_rank else 0
        if rank == 0:
            nums = [int(s) for s in msg.split() if s.isdigit()]
            for n in nums:
                if n > 1000: rank = n; break
        
        if rank == 0:
            return "Please provide your Rank to predict admission chances."

        # Gender
        gender = "M" if "male" in msg or "boy" in msg else ("F" if "female" in msg or "girl" in msg else None)
        
        # Category
        category = "OC"
        cats = ['oc_ews', 'bc_a', 'bc_b', 'bc_c', 'bc_d', 'bc_e', 'oc', 'sc', 'st']
        for c in cats:
            if c in msg.replace('-', '_'):
                category = c.upper()
                break
        if "ews" in msg and category == "OC": category = "OC_EWS"

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
        
        if not found_branch:
             return f"Please specify the branch (e.g., CSE, ECE, AIML). I understood Rank: {rank}"
             
        if not gender:
             return f"Please specify your gender (Male/Female)."

        # Predict using Pipeline (DataFrame Input)
        input_df = pd.DataFrame([{
            'Rank': rank,
            'Branch': found_branch,
            'Gender': gender,
            'Category': category
        }])
        
        prob = admission_model.predict_proba(input_df)[0][1]
        percent = prob * 100
        
        status = "High Chance" if percent > 70 else ("Moderate Chance" if percent > 40 else "Low Chance")
        
        return (f"**Admission Prediction** (Logistic Regression)\n"
                f"Details: Rank {rank}, {found_branch}, {gender}, {category}\n"
                f"Probability of Admission: **{percent:.1f}%**\n"
                f"Status: **{status}**")

    except Exception as e:
        return f"Error in prediction: {e}"

def get_intent(message):
    if not intent_model:
        return "UNKNOWN"
    return intent_model.predict([message])[0]

def respond(message, history):
    if not message:
        return ""
    
    msg_lower = message.lower()
    
    if "intake" in msg_lower or "seats" in msg_lower or "capacity" in msg_lower:
        return get_intake_info(message)
        
    # Admission Process Override (Fix for intent misclassification)
    # Force search for "Admissions Process" to get the right KB chunk
    if ("admission" in msg_lower and ("process" in msg_lower or "procedure" in msg_lower)) or "how to join" in msg_lower or "eligibility" in msg_lower:
         answer = kb.search("Admissions Process B.Tech M.Tech Eligibility") 
         return answer if answer else "Admission is determined by EAPCET rank for B.Tech and GATE/PGECET for M.Tech."
         
    # Course Info Override (Fix for M.Tech/B.Tech lookup issues)
    if "mtech" in msg_lower or "m.tech" in msg_lower:
         return ("**M.Tech Program (Master of Technology)**:\n"
                 "Pragati Engineering College offers a 2-year M.Tech (MTech) postgraduate program in specialized engineering fields. "
                 "The program is AICTE-approved and affiliated with JNTU Kakinada, focusing on advanced technical knowledge, research skills, "
                 "and industry-oriented expertise. Branches: CSE, VLSI Design, Power Electronics, Structural Engineering.")
         
    if "btech" in msg_lower or "b.tech" in msg_lower:
         return ("**About B.Tech (Bachelor of Technology)**:\n"
                 "Pragati Engineering College offers a 4-year B.Tech (BTech) program in multiple engineering disciplines. "
                 "The curriculum is AICTE-approved and affiliated with JNTU Kakinada. It focuses on technical foundations, practical skills, and industry readiness. "
                 "Available branches: CSE, AIML, Data Science, AI, Cyber Security, IT, ECE, EEE, Mechanical, Civil.")
    
    # Branch Info Override (Fix for "tell me about ds in pragati..." noise)
    # Detects branch keywords and searches specifically for that branch description
    branch_lookup = {
        "aiml": "B.Tech AIML",
        "human": "B.Tech AIML", # Context aware if needed
        "artificial intelligence": "B.Tech CSE (Artificial Intelligence)",
        "ai": "B.Tech CSE (Artificial Intelligence)",
        "cyber": "B.Tech CSE (Cyber Security)",
        "security": "B.Tech CSE (Cyber Security)",
        "data science": "B.Tech CSE (Data Science)",
        "ds": "B.Tech CSE (Data Science)",
        "cse": "B.Tech CSE (Computer Science & Engineering – Core)",
        "computer science": "B.Tech CSE (Computer Science & Engineering – Core)",
        "it": "B.Tech IT (Information Technology)",
        "information technology": "B.Tech IT (Information Technology)",
        "ece": "B.Tech ECE",
        "electronics": "B.Tech ECE",
        "eee": "B.Tech EEE",
        "electrical": "B.Tech EEE",
        "mech": "B.Tech ME",
        "mechanical": "B.Tech ME",
        "civil": "B.Tech CE"
    }
    
    # Check if user is asking "about" a branch
    if "about" in msg_lower or "explain" in msg_lower or "tell me" in msg_lower:
        # Sort keys by length so "data science" matches before "ds" or "cse"
        sorted_branches = sorted(branch_lookup.keys(), key=len, reverse=True)
        found_target = None
        detected_key = None
        
        for k in sorted_branches:
            # Word boundary check to avoid partial matches like 'it' in 'with'
            if re.search(r"\b" + re.escape(k) + r"\b", msg_lower):
                found_target = branch_lookup[k]
                detected_key = k
                break
        
        if found_target:
             answer = kb.search(found_target)
             if answer:
                 # Auto-append placements
                 p_branch = found_target  # reasonable placeholder for matching
                 if "AIML" in found_target: p_branch = "aiml"
                 elif "Data Science" in found_target: p_branch = "ds"
                 elif "Cyber" in found_target: p_branch = "cyber"
                 elif "Artificial Intelligence" in found_target: p_branch = "ai"
                 elif "IT" in found_target: p_branch = "it"
                 elif "ECE" in found_target: p_branch = "ece"
                 elif "EEE" in found_target: p_branch = "eee"
                 elif "ME" in found_target: p_branch = "mech"
                 elif "CE" in found_target: p_branch = "civil"
                 elif "CSE" in found_target: p_branch = "cse"
                 
                 try:
                    stats = get_placement_info(f"placements of {p_branch}")
                    if "unavailable" not in stats.lower() and "no placement records" not in stats.lower():
                         answer += f"\n\n**Placement Highlights for {p_branch.upper()}**:\n"
                         lines = stats.split('\n')
                         relevant_lines = [l for l in lines if any(k in l for k in ["Highest", "Average", "Total", "Recruiters"])]
                         answer += "\n".join(relevant_lines)
                 except: pass
                 
                 return answer

    # Typo tolerance for branches query
    if re.search(r"bra[a]*nch|cours|program", msg_lower):
         # Route to "Available Branches" in KB or Intent
         return kb.search("available branches") or "We offer CSE, AIML, DS, IT, ECE, EEE, MECH, CIVIL, CYBER SECURITY."

    # 1. ML Intent Detection
    intent = get_intent(message)
    print(f"Query: {message} -> Detected Intent: {intent}")
    
    if intent == "ADMISSION":
        # Split Admission Intent: Intake vs Process
        if any(w in msg_lower for w in ["process", "how to", "eligibility", "join", "fee", "application"]):
             # KB Search
             return kb.search(message) or "Admission is via EAPCET/ECET. Check website for details."
        else:
             return get_intake_info(message)
             
    elif intent == "RANK_PREDICTION": return predict_admission(message)
    elif intent == "PLACEMENT": return get_placement_info(message)
    elif intent == "PEOPLE": return get_people_info(message)
    elif intent == "COLLEGE_INFO":
        answer = kb.search(message)
        if answer: 
            # DYNAMIC PLACEMENT STATS INJECTION
            # If the user asks about a branch, we should also show placement stats as requested.
            # Check if answer contains branch keywords
            branch_map = {
                "AIML": "aiml", "CSE": "cse", "IT": "it", "ECE": "ece", 
                "EEE": "eee", "ME": "mech", "CE": "civil", "Data Science": "ds", 
                "Cyber Security": "cyber", "Artificial Intelligence": "ai"
            }
            
            detected_branch = None
            # Sort keys by length descending to match specific branches first (e.g. 'Cyber Security' before 'CSE')
            sorted_keys = sorted(branch_map.keys(), key=len, reverse=True)
            
            for key in sorted_keys:
                if key in answer: # Answer mentions the branch name (e.g. "B.Tech AIML")
                     detected_branch = branch_map[key]
                     break
            
            if detected_branch:
                # Fetch placement stats for this branch
                try:
                    stats = get_placement_info(f"placements of {detected_branch}")
                    # Only append if we get valid stats (not "unavailable")
                    if "unavailable" not in stats.lower() and "no placement records" not in stats.lower():
                         answer += f"\n\n**Placement Highlights for {detected_branch.upper()}**:\n"
                         lines = stats.split('\n')
                         relevant_lines = [l for l in lines if any(k in l for k in ["Highest", "Average", "Total", "Recruiters"])]
                         answer += "\n".join(relevant_lines)
                except Exception as e:
                    print(f"Error fetching dynamic stats: {e}")
            
            return f"{answer}"
            
        return "I couldn't find specific information in the college handbook."
    elif intent == "GREETING":
        return "Hello! I am your College AI Assistant. Ask me about Admissions, Placements, or Faculty."
    else:
        answer = kb.search(message, threshold=0.15)
        if answer: return f"I think this might help:\n{answer}"
        return "I'm not sure how to answer that."

# ==========================
# UI
# ==========================
demo = gr.ChatInterface(
    respond,
    title="🤖 Pragati AI Robo (ML Powered)",
    description="Powered by Logistic Regression, Naive Bayes, and TF-IDF.",
    examples=["Can I get seat with rank 40000 in CSE?", "Who is the Principal?", "Highest package?", "About college"]
)

if __name__ == "__main__":
    demo.launch(share=False)
