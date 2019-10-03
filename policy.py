
def dealer_policy(state):
    if state.best_sum() < 25:
        return "HIT"
    else:
        return "STICK"

def always_hit(state):
    return "HIT"

def always_stick(state):
    return "STICK"