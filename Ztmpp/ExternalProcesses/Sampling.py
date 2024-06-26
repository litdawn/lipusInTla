import numpy as np

from PT_generators.RL_Prunning.Conifg import config


def normalization(dist):
    lister = [float(x) for x in list(dist[0])]
    sumer = sum(lister)
    lister = [x / sumer for x in lister]
    return lister


def sampling(action_distribution, available_acts: dict, best=config.BEST):
    if best:
        id = -1
        maxvalue = 0
        i = 0
        for dis in action_distribution[0]:
            if float(dis) >= maxvalue:
                id = i
                maxvalue = float(dis)
            i += 1
        return available_acts[id]
    try:
        return np.random.choice(available_acts, p=normalization(action_distribution))
    except Exception as e:
        print("shit", e)
        raise e
