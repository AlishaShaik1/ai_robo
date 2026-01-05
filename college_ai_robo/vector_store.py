import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class CollegeKnowledgeBase:
    def __init__(self, data_file="college_data.txt"):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_path = os.path.join(self.base_dir, data_file)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.chunks = []
        self.tfidf_matrix = None
        self._load_and_vectorize()

    def _load_and_vectorize(self):
        if not os.path.exists(self.file_path):
            print(f"Warning: Knowledge base file not found at {self.file_path}")
            return

        with open(self.file_path, "r", encoding="utf-8") as f:
            full_text = f.read()
            
        # Split into chunks (paragraphs)
        # Assuming paragraphs are separated by double newlines
        raw_chunks = full_text.split('\n\n')
        self.chunks = [chunk.strip() for chunk in raw_chunks if chunk.strip()]
        
        if not self.chunks:
            print("Warning: No text chunks found in knowledge base.")
            return
            
        print(f"Loaded {len(self.chunks)} knowledge chunks.")
        
        # Vectorize
        self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)
        print("Knowledge Base Vectorized.")

    def search(self, query, top_k=1, threshold=0.1):
        if self.tfidf_matrix is None or not self.chunks:
            return "Knowledge base is empty."
            
        # Vectorize query
        query_vec = self.vectorizer.transform([query])
        
        # Calculate Cosine Similarity
        cosine_similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get Top-K matches
        # argsort returns indices of sorted array (ascending), so we take last k and reverse
        if top_k == 1:
            best_idx = np.argmax(cosine_similarities)
            best_score = cosine_similarities[best_idx]
            
            if best_score < threshold:
                return None # No relevant information found
            
            return self.chunks[best_idx]
        else:
            top_indices = cosine_similarities.argsort()[-top_k:][::-1]
            results = []
            for idx in top_indices:
                if cosine_similarities[idx] >= threshold:
                    results.append(self.chunks[idx])
            return results

if __name__ == "__main__":
    # Test
    kb = CollegeKnowledgeBase()
    q = "tell me about pragati"
    res = kb.search(q)
    print(f"Query: {q}")
    print(f"Result: {res}")
