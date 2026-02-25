#!/usr/bin/env python3
"""Diagnose V24 failed cases - identify exactly which questions/constraints fail and why."""
import json
import os
import re
import sys

os.environ['BOTCOIN_DEBUG'] = '0'

from botcoin_solver import (
    solve, DocumentParser, QuestionEngine, ConstraintEngine, 
    ArtifactGenerator, extract_company_names_from_doc, CompanyData,
    get_initials, next_prime
)

def diagnose_case(case, idx):
    """Run full diagnosis on a single failed case."""
    cid = case['challengeId']
    failed_cs = case['failed_constraints']
    our_artifact = case['our_artifact']
    doc = case['doc']
    questions = case['questions']
    constraints = case['constraints']
    companies_list = case['companies']
    
    print(f"\n{'='*80}")
    print(f"CASE {idx+1}: {cid}")
    print(f"Failed constraints: {failed_cs}")
    print(f"Our artifact: {our_artifact}")
    print(f"Doc format: {doc.split(chr(10))[0]}")
    print(f"{'='*80}")
    
    # Step 1: Parse companies
    company_names = extract_company_names_from_doc(doc)
    if not company_names:
        company_names = companies_list
        print(f"  [WARN] extract_company_names_from_doc returned empty, using challenge companies list")
    
    parser = DocumentParser(doc, company_names)
    companies = parser.parse()
    
    print(f"\n  Companies parsed: {len(companies)} / {len(companies_list)} expected")
    missing = set(companies_list) - set(companies.keys())
    if missing:
        print(f"  MISSING companies: {missing}")
    
    # Show key data for each company
    print(f"\n  Company data summary:")
    for name in sorted(companies.keys()):
        c = companies[name]
        rev_str = f"rev={c.revenue}" if c.revenue else "NO REV"
        print(f"    {name}: ceo={c.ceo_last_name or 'NONE'}, hq={c.hq_city or 'NONE'}/{c.hq_country or 'NONE'}, "
              f"sector={c.sector or 'NONE'}, {rev_str}, de={c.de_ratio}, sat={c.satisfaction}, "
              f"emp={c.employees}, founded={c.founded_year}, public={c.is_public}")
    
    # Step 2: Answer questions
    engine = QuestionEngine(companies)
    answers = []
    print(f"\n  Question answers:")
    for qi, q in enumerate(questions):
        ans = engine.answer(q)
        answers.append(ans)
        # Check if this question is referenced by any constraint
        refs = []
        for ci, ct in enumerate(constraints):
            if f"Question {qi+1}" in ct:
                refs.append(f"C{ci}")
        ref_str = f" (referenced by: {', '.join(refs)})" if refs else ""
        status = "✅" if ans else "❌ NO ANSWER"
        print(f"    Q{qi+1}: {q[:80]}...")
        print(f"         → {ans} {status}{ref_str}")
    
    # Step 3: Compute constraints
    ce = ConstraintEngine(constraints, questions, answers, companies)
    result = ce.compute_all()
    
    print(f"\n  Constraint computation results:")
    print(f"    word_count: {result.get('word_count')}")
    print(f"    required_elements: {result.get('required_elements')}")
    print(f"    equation: {result.get('equation')}")
    print(f"    acrostic: {result.get('acrostic')}")
    print(f"    forbidden_letter: {result.get('forbidden_letter')}")
    
    # Step 4: Generate artifact
    generator = ArtifactGenerator(result)
    new_artifact = generator.generate()
    passed, errors = generator.validate(new_artifact)
    
    print(f"\n  Generated artifact: {new_artifact}")
    print(f"  Validation: {'PASS' if passed else 'FAIL'}")
    if errors:
        for err in errors:
            print(f"    ❌ {err}")
    
    # Step 5: Also try solve() directly
    challenge_dict = {
        'doc': doc,
        'questions': questions,
        'constraints': constraints,
        'companies': companies_list
    }
    solve_artifact = solve(challenge_dict)
    solve_passed, solve_errors = generator.validate(solve_artifact)
    
    print(f"\n  solve() artifact: {solve_artifact}")
    print(f"  solve() validation: {'PASS' if solve_passed else 'FAIL'}")
    if solve_errors:
        for err in solve_errors:
            print(f"    ❌ {err}")
    
    # Step 6: Analyze specific failed constraints
    print(f"\n  FAILED CONSTRAINT ANALYSIS:")
    for ci in failed_cs:
        ct = constraints[ci]
        print(f"\n    C{ci}: {ct}")
        
        if ci == 0:  # word count
            words = new_artifact.split()
            print(f"      Word count: {len(words)}, expected from constraint")
        
        elif ci == 2:  # CEO last name
            # Find which question is referenced
            qref = re.search(r'Question\s+(\d+)', ct)
            if qref:
                qi = int(qref.group(1)) - 1
                ans = answers[qi]
                if ans and ans in companies:
                    c = companies[ans]
                    print(f"      Q{qi+1} answer: {ans}")
                    print(f"      CEO last name: {c.ceo_last_name or 'MISSING'}")
                    print(f"      CEO full name: {c.ceo_full_name or 'MISSING'}")
                else:
                    print(f"      Q{qi+1} answer: {ans} - NOT IN COMPANIES or None")
        
        elif ci == 3:  # country
            qref = re.search(r'Question\s+(\d+)', ct)
            if qref:
                qi = int(qref.group(1)) - 1
                ans = answers[qi]
                if ans and ans in companies:
                    c = companies[ans]
                    print(f"      Q{qi+1} answer: {ans}")
                    print(f"      HQ country: {c.hq_country or 'MISSING'}")
                else:
                    print(f"      Q{qi+1} answer: {ans} - NOT IN COMPANIES or None")
        
        elif ci == 4:  # prime
            qref = re.search(r'Question\s+(\d+)', ct)
            if qref:
                qi = int(qref.group(1)) - 1
                ans = answers[qi]
                if ans and ans in companies:
                    c = companies[ans]
                    emp_mod = c.employees % 100
                    prime = next_prime(emp_mod + 11)
                    print(f"      Q{qi+1} answer: {ans}")
                    print(f"      Employees: {c.employees}, mod100={emp_mod}, +11={emp_mod+11}, nextPrime={prime}")
                else:
                    print(f"      Q{qi+1} answer: {ans} - NOT IN COMPANIES or None")
        
        elif ci == 5:  # equation
            # Parse A and B refs
            a_ref = re.search(r'A=.*?Question\s+(\d+)', ct)
            b_ref = re.search(r'B=.*?Question\s+(\d+)', ct)
            if a_ref and b_ref:
                a_qi = int(a_ref.group(1)) - 1
                b_qi = int(b_ref.group(1)) - 1
                a_ans = answers[a_qi]
                b_ans = answers[b_qi]
                print(f"      A from Q{a_qi+1}: {a_ans}")
                print(f"      B from Q{b_qi+1}: {b_ans}")
                if a_ans and a_ans in companies:
                    c = companies[a_ans]
                    q1_rev = c.revenue[0] if c.revenue else 0
                    a_val = (q1_rev % 90) + 10
                    print(f"        A company Q1 rev: {q1_rev}, A = ({q1_rev} % 90) + 10 = {a_val}")
                if b_ans and b_ans in companies:
                    c = companies[b_ans]
                    q4_rev = c.revenue[3] if len(c.revenue) >= 4 else 0
                    b_val = (q4_rev % 90) + 10
                    print(f"        B company Q4 rev: {q4_rev}, B = ({q4_rev} % 90) + 10 = {b_val}")
        
        elif ci == 6:  # acrostic
            # Parse initials refs
            initials_refs = re.findall(r'initials\(Q(\d+)\)', ct)
            if initials_refs:
                acrostic_str = ''
                for ref in initials_refs:
                    qi = int(ref) - 1
                    ans = answers[qi]
                    init = get_initials(ans) if ans else '?'
                    acrostic_str += init
                    print(f"      Q{qi+1} answer: {ans} → initials: {init}")
                print(f"      Full acrostic source: {acrostic_str}")
                print(f"      First 8 chars: {acrostic_str[:8].upper()}")
                print(f"      Computed acrostic: {result.get('acrostic', 'NONE')}")
    
    return {
        'id': cid,
        'companies_parsed': len(companies),
        'companies_expected': len(companies_list),
        'missing_companies': list(missing),
        'unanswered': [i for i, a in enumerate(answers) if a is None],
        'errors': errors,
        'passed': passed
    }


def main():
    with open('v24_failed_cases.json') as f:
        cases = json.load(f)
    
    print(f"Loaded {len(cases)} V24 failed cases")
    
    summaries = []
    for i, case in enumerate(cases):
        summary = diagnose_case(case, i)
        summaries.append(summary)
    
    print(f"\n\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    for s in summaries:
        status = "✅" if s['passed'] else "❌"
        print(f"  {status} {s['id']}: {s['companies_parsed']}/{s['companies_expected']} companies, "
              f"unanswered={s['unanswered']}, errors={len(s['errors'])}")
        if s['missing_companies']:
            print(f"     Missing: {s['missing_companies'][:5]}")


if __name__ == '__main__':
    main()
