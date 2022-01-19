import re
from collections import OrderedDict

class BeliefParser:

    def __init__(self, domains):
        self.slotval_re = re.compile(r"(\w[\w ]*\w) = ([\w\d: |']+)")
        self.supported_domains = domains
        self.domain_re = re.compile(r'(' + '|'.join(domains) + r") {\s*([\w,= :\d|']*)\s*}", re.IGNORECASE)

    def __call__(self, raw_belief: str):
        belief = OrderedDict()
        for match in self.domain_re.finditer(raw_belief):
            domain, domain_bs = match.group(1), match.group(2)
            belief[domain] = {}
            for slot_match in self.slotval_re.finditer(domain_bs):
                slot, val = slot_match.group(1), slot_match.group(2)
                belief[domain][slot] = val
        return belief


if __name__ == '__main__':
    # usage example
    bp = BeliefParser(['hotel', 'restaurant', 'attraction', 'train', 'taxi', 'police', 'hospital'])
    # textual representation looks like this
    belief = bp('restaurant {area = centre, price = cheap} train {leave at = 9:00, departure = cambridge}')
    # output: OrderedDict([('restaurant', {'area': 'centre', 'price': 'cheap'}), ('train', {'leave at': '9:00', 'departure': 'cambridge'})])
    print(belief)
