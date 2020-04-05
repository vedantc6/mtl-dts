import json
from collections import OrderedDict
import os 

conll_data = json.load(open("./data/datasets/conll04/conll04_types.json"), object_pairs_hook=OrderedDict)

for i, (key, val) in enumerate(conll_data['entities'].items()):
    print(i, key, val)
