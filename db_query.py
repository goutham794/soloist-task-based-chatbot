from database.database import MultiWOZDatabase

import re
from collections import OrderedDict

db = MultiWOZDatabase()

class BeliefParser:

    def __init__(self, domains):
        self.slotval_re = re.compile(r"(\w[\w ]*\w) [=:] ([\w\d: |']+)")
        self.supported_domains = domains
        self.domain_re = re.compile(r'(' + '|'.join(domains) + r") {\s*([\w,= :\d|']*)\s*}", re.IGNORECASE)
        # self.domain_re = re.compile(r'(' + '|'.join(domains) + r") {\s*([\w,= :\d|']*)\s*}", re.IGNORECASE)

    def __call__(self, raw_belief: str):
        belief = OrderedDict()
        for match in self.domain_re.finditer(raw_belief):
            domain, domain_bs = match.group(1), match.group(2)
            belief[domain[:-2]] = {}
            for slot_match in self.slotval_re.finditer(domain_bs):
                slot, val = slot_match.group(1), slot_match.group(2)
                belief[domain[:-2]][slot] = val
        return belief

bp = BeliefParser(['hotel :', 'restaurant :', 'attraction :', 'train :', 'taxi :', 'police :', 'hospital :'])
bs = bp('<|belief|> {train : {day : monday, departure : cambridge, destination : ely, leaveat : 13:15}}')

bs = {"restaurant": {"area": ["centre"], "bookday": ["saturday"], "bookpeople": ["7"], "booktime": ["16:00"], "food": ["indian"], "name": ["dontcare"], "pricerange": ["expensive"]}}

bs = {"restaurant": {"area": "centre", "bookday": "saturday", "bookpeople": "7", "booktime": "16:00", "food": "indian", "name": "dontcare", "pricerange": "expensive"}}

bs = {"hotel" : {"pricerange" : "moderate", "type" : "guesthouse"}}

print(bs)

db_results = {}
for domain,constraints in bs.items():
    db_results[domain] = len(db.query(domain, constraints = constraints))

print(db_results)