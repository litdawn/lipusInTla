import time


class Timer:
    def __init__(self):
        self.timer_dict = dict()

    def new_timer(self, name):
        self.timer_dict[name] = time.time()

    def get_time(self, name):
        return time.time() - self.timer_dict[name]

    def get_and_refesh(self, name):
        all = time.time() - self.timer_dict[name]
        self.timer_dict[name] = time.time()
        return all


timer = Timer()

from enum import Enum


class TIMER(Enum):
    TOTAL = 0
    EPOCH = 1
    CTI_GENERATOR = 2
    CTI_ELIMINATOR = 3
    LEMMA_GENERATOR = 4
    LEMMA_CHECKER = 5
