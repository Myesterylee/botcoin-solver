import json

with open('test_samples.json') as f:
    samples = json.load(f)

for i, sample in enumerate(samples):
    print(f"\n=== SAMPLE {i} ===")
    print(f"Artifact: {sample.get('our_artifact', 'N/A')}")
    print(f"Failed constraints: {sample.get('failed_constraints', [])}")
    print(f"Constraint texts: {sample.get('constraint_texts', [])}")
    
    # Also print the questions and constraints from the challenge
    challenge = sample.get('challenge', {})
    questions = challenge.get('questions', [])
    constraints = challenge.get('constraints', [])
    companies = challenge.get('companies', [])
    
    print(f"\nQuestions ({len(questions)}):")
    for j, q in enumerate(questions):
        print(f"  Q{j+1}: {q}")
    
    print(f"\nConstraints ({len(constraints)}):")
    for j, c in enumerate(constraints):
        print(f"  C{j+1}: {c}")
    
    print(f"\nCompanies ({len(companies)}): {companies}")
