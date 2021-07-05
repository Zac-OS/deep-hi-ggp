# store all known information for each player + no probabilities
# also store true state for game runner
# have a generator for possible states based on this players knowledge

class ImperfectNode(PerfectNode):
    def __init__(self, propnet, data):
        super().__init__(propnet, data)
