from app import respond, init_placement_data, load_admission_data

print("--- Re-Initializing Data ---")
init_placement_data()
load_admission_data()

# Verification
q = "how many placements for ai"
print(f"\nQuery: {q}")
try:
    ans = respond(q, [])
    print(f"Answer: {ans}")
except Exception as e:
    print(f"Error: {e}")
