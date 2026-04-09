import json
from eudaimonia.server.app import info

res = info()
print(json.dumps(res, indent=2))
