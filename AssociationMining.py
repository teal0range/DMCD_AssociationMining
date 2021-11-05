from abc import abstractmethod
from collections import defaultdict
import time
import numpy as np
from FormatReader import *
import os
from utils import *

logger = getLogger(__file__)


def timer(f):
    def inner(*args, **kwargs):
        st = time.time()
        _res = f(*args, **kwargs)
        logger.info(f"function {f.__name__} cost {time.time() - st}s")
        return _res

    inner.__name__ = f.__name__
    return inner


class Algorithm:
    def __init__(self, support, confidence):
        self.support = support
        self.confidence = confidence
        self.count = None

    @abstractmethod
    def run(self, dataset):
        pass


class Dummy(Algorithm):

    def run(self, dataset):
        pass


class Apriori(Algorithm):

    def run(self, dataset):
        self.count = len(dataset)
        genCount, dataset = self.init_generation(dataset)
        while len(genCount) != 0:
            genCount, dataset = self.iteration(dataset, genCount)

    @timer
    def iteration(self, dataset, genCount):
        gen = self.prune(list(genCount.keys()))
        genCount, dataset = self.countFreq(gen, dataset)
        return genCount, dataset

    @timer
    def init_generation(self, dataset):
        genCount = defaultdict(lambda: 0)
        next_dataset = set()
        cnt = 0
        for idx, itemset in enumerate(dataset):
            for item in itemset:
                genCount[(item,)] += 1
                cnt += 1
                if genCount[(item,)] > self.count * self.support:
                    next_dataset.add(idx)
        next_dataset = [set(dataset[idx]) for idx in next_dataset]
        genCount = {key: val for key, val in genCount.items() if val > self.count * self.support}
        return genCount, next_dataset

    def countFreq(self, gen, dataset):
        genCount = defaultdict(lambda: 0)
        next_dataset = set()
        for itemset in gen:
            for idx, data in enumerate(dataset):
                if set(itemset) <= data:
                    genCount[tuple(itemset)] += 1
                    if genCount[tuple(itemset)] > self.count * self.support:
                        next_dataset.add(idx)
        next_dataset = [dataset[idx] for idx in next_dataset]
        genCount = {key: val for key, val in genCount.items() if val > self.count * self.support}
        return genCount, next_dataset

    @staticmethod
    def prune(freq_set):
        nextGen = set()
        # nextGenCount = defaultdict(lambda: 0)
        for item_setx in freq_set:
            for item_sety in freq_set:
                isEqual = [item_sety[i] == item_sety[i] for i in range(len(item_setx))]
                if np.prod(isEqual[:-1]) and item_setx[-1] < item_sety[-1]:
                    nextGen.add(item_setx + (item_sety[-1],))
        return nextGen


class FPGrowth(Algorithm):

    def run(self, dataset):
        pass


if __name__ == '__main__':
    Apriori(0.01, 0.2).run(UnixReader().read())
