import json
with open('validate_url_output.json', encoding='utf-16le') as f:
    content = f.read().replace('\ufeff', '')

data = json.loads(content)
for t in data.get('criteria', []):
    if not t.get('passed'):
        print(f"FAILED: {t['id']}")
        print(f"Expected: {json.dumps(t.get('expected', {}), indent=2)}")
        print(f"Actual: {json.dumps(t.get('actual', {}), indent=2)}")
        print("-" * 40)
