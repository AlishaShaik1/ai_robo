import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "intent_model.pkl")

# Labeled Dataset
# Classes: PEOPLE, PLACEMENT, ADMISSION, RANK_PREDICTION, COLLEGE_INFO, UNKNOWN
# Added more examples based on feedback
data = [
    # RANK_PREDICTION
    ("can i get a seat with rank 30000", "RANK_PREDICTION"),
    ("my rank is 45000 will i get cse", "RANK_PREDICTION"),
    ("chances for rank 10000 in ece", "RANK_PREDICTION"),
    ("predict my admission probability", "RANK_PREDICTION"),
    ("is my rank good enough for aiml", "RANK_PREDICTION"),
    ("rank 5000 oc male", "RANK_PREDICTION"),
    ("i got 20000 rank in eapcet", "RANK_PREDICTION"),
    ("cutoff rank for cse", "RANK_PREDICTION"),
    ("last year cutoff for it", "RANK_PREDICTION"),
    ("am i eligible for mechanical", "RANK_PREDICTION"),
    ("rank predictor", "RANK_PREDICTION"),
    ("can i get seat", "RANK_PREDICTION"),
    ("rank 40000 in cse", "RANK_PREDICTION"),
    ("rank 100000 in cse", "RANK_PREDICTION"),
    ("seat with rank", "RANK_PREDICTION"),

    # ADMISSION (Stats/Intake)
    ("what is the intake of cse", "ADMISSION"),
    ("how many seats in aiml", "ADMISSION"),
    ("total seats availabe", "ADMISSION"),
    ("number of students in ece", "ADMISSION"),
    ("admission process", "ADMISSION"),
    ("how many join civil", "ADMISSION"),
    ("seat matrix", "ADMISSION"),
    ("intake capacity", "ADMISSION"),
    ("what is intake of aiml", "ADMISSION"),
    ("intake of it", "ADMISSION"),
    ("seats in cse", "ADMISSION"),
    
    # PLACEMENT (Specific & General)
    ("how are placements here", "PLACEMENT"),
    ("highest package this year", "PLACEMENT"),
    ("average salary for cse", "PLACEMENT"),
    ("top recruiters", "PLACEMENT"),
    ("companies visiting college", "PLACEMENT"),
    ("job opportunities", "PLACEMENT"),
    ("placement statistics", "PLACEMENT"),
    ("did tcs come", "PLACEMENT"),
    ("what is the lowest package", "PLACEMENT"),
    ("placement percentage", "PLACEMENT"),
    ("jobs after aiml", "PLACEMENT"),
    ("what are the placements of aiml", "PLACEMENT"),
    ("what are the placements of cse", "PLACEMENT"),
    ("placements info", "PLACEMENT"),
    ("placements for ece", "PLACEMENT"),
    ("how many verified placements", "PLACEMENT"),
    ("count of placed students", "PLACEMENT"),
    ("salary pacakge", "PLACEMENT"),

    # PEOPLE
    ("who is the principal", "PEOPLE"),
    ("name of the chairman", "PEOPLE"),
    ("who is hod of cse", "PEOPLE"),
    ("director name", "PEOPLE"),
    ("dean of academics", "PEOPLE"),
    ("tell me about dr naresh", "PEOPLE"),
    ("placement officer info", "PEOPLE"),
    ("who is krishna rao", "PEOPLE"),
    ("faculty list", "PEOPLE"),
    ("who is hod of aiml", "PEOPLE"),
    ("who is hod of it", "PEOPLE"),
    ("who created you", "PEOPLE"),
    ("creator name", "PEOPLE"),
    ("hod name", "PEOPLE"),

    # COLLEGE_INFO
    ("tell me about the college", "COLLEGE_INFO"),
    ("where is the college located", "COLLEGE_INFO"),
    ("college address", "COLLEGE_INFO"),
    ("history of pragati", "COLLEGE_INFO"),
    ("vision and mission", "COLLEGE_INFO"),
    ("is there hostel facility", "COLLEGE_INFO"),
    ("bus transport availability", "COLLEGE_INFO"),
    ("contact details", "COLLEGE_INFO"),
    ("phone number of office", "COLLEGE_INFO"),
    ("facilities in college", "COLLEGE_INFO"),
    ("explain about pragati engineering college", "COLLEGE_INFO"),
    ("available branches", "COLLEGE_INFO"), # "what branches are there"

    # GREETING
    ("hi", "GREETING"),
    ("hello", "GREETING"),
    ("good morning", "GREETING"),
    ("thank you", "GREETING"),
    ("bye", "GREETING"),
]

def train_intent_model():
    print("Training Intent Detection Model (Naive Bayes)...")
    
    df = pd.DataFrame(data, columns=['text', 'intent'])
    
    # Preprocessing
    X = df['text']
    y = df['intent']
    
    # Train Full Data (No Split for Production to maximize utility of small dataset)
    # But for verification we can print metrics on a subset, or just train on all since data is tiny.
    # Let's train on ALL data for the final model.
    
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), MultinomialNB())
    model.fit(X, y)
    
    print("Model trained on full dataset.")

    # Sanity Checks
    test_phrases = [
        "rank 40000 cse",
        "who is the principal",
        "highest package",
        "about college",
        "who is hod of aiml",
        "placements of cse"
    ]
    print("\nSanity Checks:")
    for phrase in test_phrases:
        pred = model.predict([phrase])[0]
        print(f"'{phrase}' -> {pred}")

    # Save
    joblib.dump(model, MODEL_PATH)
    print(f"Intent Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_intent_model()
