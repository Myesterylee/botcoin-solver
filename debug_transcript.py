import json, sys
sys.path.insert(0, '.')
from botcoin_solver import DocumentParser

with open('test_samples.json') as f:
    samples = json.load(f)
sample = samples[4]
challenge = sample['challenge']

parser = DocumentParser(challenge['doc'], challenge['companies'])
parser._pass_identity()
parser._build_name_lookup()

print('Name lookup keys after identity + build:')
for k in sorted(parser.name_lookup.keys()):
    print(f'  {k!r} -> {parser.name_lookup[k]!r}')

print()
print('Testing resolve for key names:')
for name in ['GC', 'gc', 'Giga Cloud', 'giga cloud', 'NA', 'QL', 'QQ', 'CH', 'AN', 'XW', 'BQ', 'VG', 'OF', 'XN', 'AS', 'LA', 'ZC', 'AV']:
    result = parser._resolve_company(name)
    print(f'  resolve({name!r}) -> {result!r}')
