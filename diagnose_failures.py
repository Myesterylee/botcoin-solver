#!/usr/bin/env python3
"""Diagnose why certain challenges fail in batch test."""
import json, random, os, sys
os.environ['BOTCOIN_DEBUG'] = '0'
from botcoin_solver import solve, extract_company_names_from_doc, DocumentParser, QuestionEngine, ConstraintEngine, ArtifactGenerator

all_ch = []
with open('challenges.jsonl') as f:
    for line in f:
        if line.strip():
            all_ch.append(json.loads(line))

random.seed(42)
selected = random.sample(all_ch, 20)

fail_indices = [0, 5, 6, 9, 13, 14, 16, 17]
for idx in fail_indices:
    ch = selected[idx]
    doc = ch.get('doc', '')
    companies_list = ch.get('companies', [])
    company_names = extract_company_names_from_doc(doc)
    parser = DocumentParser(doc, company_names)
    companies = parser.parse()
    
    print(f"=== Sample {idx+1} (epoch={ch.get('epochId')}) ===")
    print(f"  companies field: {companies_list[:5]}...")
    print(f"  extracted names: {company_names[:5]}...")
    print(f"  parsed companies: {len(companies)}")
    if companies:
        for name, c in list(companies.items())[:3]:
            print(f"    {name}: rev={c.revenue}, hq={c.hq_city}/{c.hq_country}")
    print(f"  doc first 400 chars:")
    print(f"  {doc[:400]}")
    print()
    
    # Also show the artifact
    artifact = solve(ch)
    print(f"  artifact: {repr(artifact)}")
    print()
