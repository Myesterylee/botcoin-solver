#!/usr/bin/env python3
"""Categorize unanswered questions to find patterns."""
import json, random, os, re
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

# Categorize by keywords
categories = Counter()
for q in unanswered:
    ql = q.lower()
    if 'ceo' in ql and ('initial' in ql or 'first letter' in ql or 'starts with' in ql):
        categories['CEO_INITIAL_MATCH'] += 1
    elif 'satisfaction' in ql and 'growth' in ql and ('combined' in ql or 'adding' in ql or 'plus' in ql or 'sum' in ql or 'score' in ql):
        categories['SAT_PLUS_GROWTH'] += 1
    elif 'volatility' in ql or ('gap' in ql and 'quarter' in ql and 'revenue' in ql):
        categories['VOLATILITY'] += 1
    elif 'per' in ql and ('employee' in ql or 'headcount' in ql) and 'revenue' in ql:
        categories['EMP_PER_REV'] += 1
    elif 'total' in ql and 'revenue' in ql:
        categories['TOTAL_REV'] += 1
    elif 'average' in ql and 'growth' in ql:
        categories['AVG_GROWTH'] += 1
    elif 'second' in ql and ('oldest' in ql or 'earliest' in ql):
        categories['SECOND_OLDEST'] += 1
    elif 'ipo' in ql or 'went public' in ql:
        categories['IPO'] += 1
    elif ('public' in ql or 'traded' in ql) and ('d/e' in ql or 'debt' in ql):
        categories['PUBLIC_DE'] += 1
    elif ('private' in ql or 'privately' in ql) and 'satisfaction' in ql:
        categories['PRIVATE_SAT'] += 1
    elif ('private' in ql or 'privately' in ql) and ('employee' in ql or 'headcount' in ql):
        categories['PRIVATE_EMP'] += 1
    elif ('q3' in ql or 'third quarter' in ql) and ('decline' in ql or 'negative' in ql or 'drop' in ql):
        categories['Q3_DECLINE_DE'] += 1
    elif re.search(r'(?:over|more than)\s+[\d,]+\s+employee', ql) and 'revenue' in ql:
        categories['OVER_EMP_REV'] += 1
    elif 'billion' in ql and 'satisfaction' in ql:
        categories['OVER_1B_SAT'] += 1
    elif 'sector' in ql and ('oldest' in ql or 'earliest' in ql):
        categories['SECTOR_OLDEST'] += 1
    elif ('q2' in ql or 'second quarter' in ql or 'second-quarter' in ql) and 'revenue' in ql:
        categories['CITY_Q2'] += 1
    elif ('headquartered' in ql or 'hq' in ql or 'based in' in ql) and ('employee' in ql or 'employs' in ql):
        categories['LOC_EMP'] += 1
    elif 'founded' in ql and ('before' in ql or 'earlier' in ql or 'prior' in ql) and ('q4' in ql or 'fourth' in ql or 'closing' in ql):
        categories['BEFORE_Q4'] += 1
    elif ('q1' in ql or 'first quarter' in ql) and ('q4' in ql or 'fourth quarter' in ql):
        categories['Q1_Q4_INCREASE'] += 1
    elif re.search(r'the\s+\d{4}s', ql) and 'positive' in ql and 'growth' in ql:
        categories['DECADE_POSITIVE'] += 1
    elif 'public' in ql and 'positive' in ql and 'growth' in ql and ('fewest' in ql or 'least' in ql or 'smallest' in ql):
        categories['PUBLIC_POS_FEW_EMP'] += 1
    elif ('d/e' in ql or 'debt' in ql) and 'satisfaction' in ql:
        categories['DE_SAT_Q4'] += 1
    else:
        categories['UNKNOWN'] += 1

print(f"Total unanswered: {len(unanswered)}")
print(f"\nCategories:")
for cat, count in categories.most_common():
    print(f"  {cat}: {count}")

# Show examples of UNKNOWN
print(f"\n=== UNKNOWN examples ===")
unknowns = [q for q in unanswered if True]  # we'll filter below
count = 0
for q in unanswered:
    ql = q.lower()
    # Check if it falls into UNKNOWN
    matched = False
    for kw_set in [
        ('ceo', 'initial'), ('satisfaction', 'growth', 'combined'),
        ('volatility',), ('per', 'employee', 'revenue'),
        ('total', 'revenue'), ('average', 'growth'),
        ('second', 'oldest'), ('ipo',), ('went public',),
        ('public', 'd/e'), ('public', 'debt'),
        ('private', 'satisfaction'), ('privately', 'satisfaction'),
        ('private', 'employee'), ('privately', 'employee'),
        ('q3', 'decline'), ('third quarter', 'decline'),
        ('billion', 'satisfaction'), ('sector', 'oldest'),
        ('q2', 'revenue'), ('second quarter', 'revenue'),
        ('headquartered', 'employee'), ('hq', 'employee'),
        ('founded', 'before', 'q4'), ('founded', 'earlier', 'closing'),
        ('q1', 'q4'), ('first quarter', 'fourth quarter'),
        ('positive', 'growth', 'fewest'),
        ('d/e', 'satisfaction'), ('debt', 'satisfaction'),
    ]:
        if all(k in ql for k in kw_set):
            matched = True
            break
    if not matched:
        count += 1
        if count <= 30:
            print(f"  [{count}] {q}")