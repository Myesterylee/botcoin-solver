#!/usr/bin/env python3
"""Batch test botcoin_solver against the full dataset."""

import json
import random
import sys
import time
import os

# Suppress debug output
os.environ['BOTCOIN_DEBUG'] = '0'

from botcoin_solver import solve, DocumentParser, QuestionEngine, ConstraintEngine, ArtifactGenerator, extract_company_names_from_doc

def load_challenges(path='challenges.jsonl', n=20, seed=42):
    """Load n random challenges from JSONL file."""
    all_challenges = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                all_challenges.append(json.loads(line))
    
    random.seed(seed)
    selected = random.sample(all_challenges, min(n, len(all_challenges)))
    return selected

def validate_artifact(challenge, artifact):
    """Validate artifact against challenge constraints using the solver's own engine."""
    doc = challenge.get('doc', '')
    questions = challenge.get('questions', [])
    constraints = challenge.get('constraints', [])
    
    company_names = extract_company_names_from_doc(doc)
    parser = DocumentParser(doc, company_names)
    companies = parser.parse()
    
    engine = QuestionEngine(companies)
    answers = []
    for q in questions:
        ans = engine.answer(q)
        answers.append(ans)
    
    constraint_engine = ConstraintEngine(constraints, questions, answers, companies)
    constraint_info = constraint_engine.compute_all()
    
    generator = ArtifactGenerator(constraint_info)
    passed, errors = generator.validate(artifact)
    return passed, errors, len(constraints)

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    
    print(f"Loading challenges from challenges.jsonl...")
    challenges = load_challenges('challenges.jsonl', n=n, seed=seed)
    print(f"Selected {len(challenges)} challenges (seed={seed})\n")
    
    total_constraints = 0
    total_passed = 0
    total_failed = 0
    perfect_count = 0
    results = []
    
    for i, ch in enumerate(challenges):
        cid = ch.get('id', 'unknown')[:16]
        epoch = ch.get('epochId', '?')
        n_questions = len(ch.get('questions', []))
        n_constraints = len(ch.get('constraints', []))
        
        print(f"[{i+1:2d}/{len(challenges)}] id={cid}... epoch={epoch} q={n_questions} c={n_constraints}", end=" ")
        sys.stdout.flush()
        
        t0 = time.time()
        try:
            artifact = solve(ch)
            elapsed = time.time() - t0
            
            passed, errors, _ = validate_artifact(ch, artifact)
            n_passed = n_constraints - len(errors)
            total_constraints += n_constraints
            total_passed += n_passed
            total_failed += len(errors)
            
            if passed:
                perfect_count += 1
                print(f"✅ {n_passed}/{n_constraints} ({elapsed:.1f}s)")
            else:
                print(f"❌ {n_passed}/{n_constraints} ({elapsed:.1f}s)")
                for err in errors:
                    print(f"      → {err}")
            
            results.append({
                'id': ch.get('id', ''),
                'epoch': epoch,
                'passed': n_passed,
                'total': n_constraints,
                'perfect': passed,
                'errors': errors,
                'time': elapsed
            })
        except Exception as e:
            elapsed = time.time() - t0
            print(f"💥 ERROR ({elapsed:.1f}s): {e}")
            results.append({
                'id': ch.get('id', ''),
                'epoch': epoch,
                'passed': 0,
                'total': n_constraints,
                'perfect': False,
                'errors': [str(e)],
                'time': elapsed
            })
            total_constraints += n_constraints
            total_failed += n_constraints
    
    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Challenges tested:  {len(challenges)}")
    print(f"Perfect (all pass): {perfect_count}/{len(challenges)} ({100*perfect_count/len(challenges):.1f}%)")
    print(f"Constraints:        {total_passed}/{total_constraints} ({100*total_passed/total_constraints:.1f}%)")
    print(f"Avg time/challenge: {sum(r['time'] for r in results)/len(results):.2f}s")
    
    # Per-epoch breakdown
    epochs = {}
    for r in results:
        ep = str(r['epoch'])
        if ep not in epochs:
            epochs[ep] = {'passed': 0, 'total': 0, 'perfect': 0, 'count': 0}
        epochs[ep]['passed'] += r['passed']
        epochs[ep]['total'] += r['total']
        epochs[ep]['perfect'] += int(r['perfect'])
        epochs[ep]['count'] += 1
    
    print(f"\nPer-epoch breakdown:")
    for ep in sorted(epochs.keys()):
        e = epochs[ep]
        print(f"  Epoch {ep}: {e['perfect']}/{e['count']} perfect, {e['passed']}/{e['total']} constraints ({100*e['passed']/e['total']:.1f}%)")

if __name__ == '__main__':
    main()
