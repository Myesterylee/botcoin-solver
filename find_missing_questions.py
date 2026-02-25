#!/usr/bin/env python3
"""Find questions that the solver can't answer to identify missing question types."""
import json, random, os, sys
from collections import Counter
os.environ['BOTCOIN_DEBUG'] = '0'
from botcoin_solver import extract_company_names_from_doc, DocumentParser, QuestionEngine

all_ch = []
with open('challenges.jsonl') as f:
    for line in f:
        if line.strip():
            all_ch.append(json.loads(line))

random.seed(42)
selected = random.sample(all_ch, 300)

unanswered = []
total_q = 0
answered_q = 0

for ch in selected:
    doc = ch.get('doc', '')
    questions = ch.get('questions', [])
    company_names = extract_company_names_from_doc(doc)
    if not company_names:
        company_names = ch.get('companies', [])
    parser = DocumentParser(doc, company_names)
    companies = parser.parse()
    if not companies:
        continue
    engine = QuestionEngine(companies)
    for q in questions:
        total_q += 1
        ans = engine.answer(q)
        if ans is None:
            unanswered.append(q)
        else:
            answered_q += 1

print(f"Total questions: {total_q}")
print(f"Answered: {answered_q} ({100*answered_q/total_q:.1f}%)")
print(f"Unanswered: {len(unanswered)} ({100*len(unanswered)/total_q:.1f}%)")
print(f"\n=== UNANSWERED QUESTIONS (sample) ===")

# Group by similarity - show unique patterns
seen = set()
unique = []
for q in unanswered:
    # Normalize for dedup
    key = q.lower()[:80]
    if key not in seen:
        seen.add(key)
        unique.append(q)

print(f"\nUnique patterns: {len(unique)}")
for i, q in enumerate(unique[:50]):
    print(f"  [{i+1}] {q}")
