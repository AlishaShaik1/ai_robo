import os
import difflib
import re

# Mocking the data loading part from app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
people_file = os.path.join(BASE_DIR, "people_data.txt")
PEOPLE_DICT = {}
valid_roles = []

print(f"Reading {people_file}...")
if os.path.exists(people_file):
    with open(people_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            if ":" in line:
                parts = line.split(":", 1)
                role = parts[0].strip()
                name = parts[1].strip()
                PEOPLE_DICT[role.lower()] = name
                valid_roles.append(role.lower())
                print(f"Loaded: '{role.lower()}' -> '{name}'")

def correct_typos(message):
    # Common mappings (manual overrides)
    custom_corrections = {
        "princpaaal": "principal",
        "chaman": "chairman",
        "hd": "hod",
        "deaan": "dean",
        "princi": "principal",
        "hod aiml": "hod aiml",
        "who created you": "creator",
        "who made you": "creator",
        "created by": "creator"
    }
    
    # Pre-process message to handle multi-word aliases for HODs *before* splitting
    msg_norm = message.lower()
    
    # HOD Alias Map
    hod_aliases = {
        "data science": "hod data science",
        "cse(ds)": "hod data science",
        "cse (ds)": "hod data science",
        "cse ds": "hod data science",
        "ds": "hod data science",
        
        "artificial intelligence": "hod ai",
        "cse(ai)": "hod ai",
        "cse (ai)": "hod ai",
        "ai": "hod ai",
        
        "cyber security": "hod cyber security",
        "cyber": "hod cyber security",
        "cse(cs)": "hod cyber security",
        "csc": "hod cyber security",
        
        "aiml": "hod aiml",
        "cse(aiml)": "hod aiml",
        
        "cse(ai&ml)": "hod aiml",
        "cse (ai&ml)": "hod aiml",
        "cse ai&ml": "hod aiml",
        "ai&ml": "hod aiml",

        "it": "hod it",
        "cse(it)": "hod it",
        "information technology": "hod it",
        
        "cse": "hod cse",
        "computer science": "hod cse",
        "cs": "hod cse",
        
        "ece": "hod ece",
        "electronics": "hod ece",
        
        "eee": "hod eee",
        "electrical": "hod eee",
        
        "mec": "hod mechanical",
        "mech": "hod mechanical",
        "mechanical": "hod mechanical",
        
        "civil": "hod civil",
        "civ": "hod civil"
    }

    import re
    # Only apply HOD aliases if "hod" is likely implied or user asks "who is hod ..."
    # Use Regex with Lookaround for boundaries to handle parens like "cse(aiml)"
    
    # Sort aliases by length (descending) 
    sorted_aliases = sorted(hod_aliases.items(), key=lambda x: len(x[0]), reverse=True)
    
    for alias, target in sorted_aliases:
        # Construct patterns using lookarounds instead of \b which fails on symbols
        # (?<!\w) matches if not preceded by a word char
        # (?!\w) matches if not followed by a word char
        esc_alias = re.escape(alias)
        
        patterns = [
            rf"(?<!\w)hod\s+{esc_alias}(?!\w)", 
            rf"(?<!\w)hod\s+of\s+{esc_alias}(?!\w)", 
            rf"(?<!\w)head\s+of\s+{esc_alias}(?!\w)"
        ]
        for p in patterns:
            msg_norm = re.sub(p, target, msg_norm)
    
    # Reverse check: "ds hod"
    if "hod" in msg_norm:
         for alias, target in sorted_aliases:
             esc_alias = re.escape(alias)
             p = rf"(?<!\w){esc_alias}\s+hod(?!\w)"
             msg_norm = re.sub(p, target, msg_norm)

    words = msg_norm.split()
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

# Mock Respond Logic for Name Lookup
def respond_mock(message):
    corrected_message = correct_typos(message)
    lower_msg = corrected_message.lower()
    
    for role, name in PEOPLE_DICT.items():
        if role in lower_msg:
             return f"The {role.title()} is {name}."
    return "LLM Fallback"

# Test cases
tests = [
    "hod of cse(aiml)",
    "hod of cse(ds)",
    "hod of cse(ai)",
    "who is hod of ece",
    "hod of cse(ai&ml)", # Need to add this alias to the map manually first for this test
]

# Quick inject alias for test
PEOPLE_DICT["hod aiml"] = "Dr. Radha Krishna Sir" # ensure mock data exists

# Inject cse(ai&ml) into aliases context if missing in original copy?
# The original copy above has hardcoded list. I need to update it too.
# I'll rely on the fact that I'm re-writing the function logic, 
# but I need to update the dict passed to sorted() implicitly?
# Actually `hod_aliases` is defined inside correct_typos in the file.
# I need to update that list in the replacement too? 
# The replacement chunk didn't include dict.
# I should include dict update if I want to test AI&ML.
