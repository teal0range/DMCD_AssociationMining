import time
import sys
import traceback

from FormatReader import *
from utils import *

logger = getLogger(__file__)


class timer:
    call_level = 0

    def __init__(self, log=False):
        self.log = log

    def __call__(self, f):
        def inner(*args, **kwargs):
            st = time.time()
            if self.log:
                logger.info("\t" * self.call_level + f"function {f.__name__} started")
            timer.call_level += 1
            _res = f(*args, **kwargs)
            timer.call_level -= 1
            if self.log:
                logger.info("\t" * self.call_level + f"function {f.__name__} finished, cost {time.time() - st}s")
            return _res

        inner.__name__ = f.__name__
        return inner


class Algorithm:
    def __init__(self, support, confidence):
        self.support = support
        self.confidence = confidence
        self.itemsetCount = {}
        self.count = None
        self.associationMiner = None

    @abstractmethod
    def run(self, dataset):
        pass


class Dummy(Algorithm):
    @timer()
    def run(self, dataset):
        self.count = len(dataset)
        self.associationMiner = AssociationRuleMiner(self.support, self.confidence, self.count)
        genCount = self.init_generation(dataset)
        self.itemsetCount.update(genCount)
        while len(genCount) != 0:
            genCount = self.iteration(dataset, genCount)
            self.itemsetCount.update(genCount)
        self.associationMiner.association_rules(list(self.itemsetCount.keys()), self.itemsetCount)

    @timer()
    def iteration(self, dataset, genCount):
        candidate = self.brute_force(list(genCount.keys()))
        genCount = self.countFreq(candidate, dataset)
        return genCount

    @timer()
    def init_generation(self, dataset):
        genCount = defaultdict(lambda: 0)
        for idx, itemset in enumerate(dataset):
            for item in itemset:
                genCount[(item,)] += 1
        genCount = {key: val for key, val in genCount.items() if val > self.count * self.support}
        return genCount

    @staticmethod
    @timer()
    def brute_force(genCount):
        next_gen = []
        items = set([_ for gen in genCount for _ in gen])
        for candidate in genCount:
            for item in items:
                if item > candidate[-1]:
                    next_gen.append(tuple(candidate) + tuple([item]))
        return sorted(set(next_gen))

    @timer()
    def countFreq(self, candidates, dataset):
        genCount = defaultdict(lambda: 0)
        for itemset in candidates:
            item_tuple = tuple(itemset)
            item_set = set(itemset)
            for idx, data in enumerate(dataset):
                if item_set <= data:
                    genCount[item_tuple] += 1
        genCount = {key: val for key, val in genCount.items() if val > self.count * self.support}
        return genCount


class Apriori(Algorithm):
    @timer()
    def run(self, dataset):
        self.count = len(dataset)
        self.associationMiner = AssociationRuleMiner(self.support, self.confidence, self.count)
        genCount, dataset = self.init_generation(dataset)
        self.itemsetCount.update(genCount)
        while len(genCount) != 0:
            genCount, dataset = self.iteration(dataset, genCount)
            self.itemsetCount.update(genCount)
        self.associationMiner.association_rules(list(self.itemsetCount.keys()), self.itemsetCount)

    @timer()
    def iteration(self, dataset, genCount):
        candidate = self.prune(sorted(list(genCount.keys())))
        genCount, dataset = self.countFreq(candidate, dataset)
        return genCount, dataset

    @timer()
    def init_generation(self, dataset):
        genCount = defaultdict(lambda: 0)
        next_dataset = set()
        for idx, itemset in enumerate(dataset):
            for item in itemset:
                genCount[(item,)] += 1
                next_dataset.add(idx)
        next_dataset = [dataset[idx] for idx in next_dataset]
        genCount = {key: val for key, val in genCount.items() if val > self.count * self.support}
        return genCount, next_dataset

    @timer()
    def countFreq(self, candidates, dataset):
        genCount = defaultdict(lambda: 0)
        next_dataset = set()
        for itemset in candidates:
            item_tuple = tuple(itemset)
            item_set = set(itemset)
            for idx, data in enumerate(dataset):
                if item_set <= data:
                    genCount[item_tuple] += 1
                    next_dataset.add(idx)
        next_dataset = [dataset[idx] for idx in next_dataset]
        genCount = {key: val for key, val in genCount.items() if val > self.count * self.support}
        return genCount, next_dataset

    @staticmethod
    @timer()
    def prune(freq_set):
        nextGen = set()
        # nextGenCount = defaultdict(lambda: 0)
        for idx, item_setx in enumerate(freq_set):
            for item_sety in freq_set[idx + 1:]:
                flag = True
                for i in range(len(item_setx) - 2, -1, -1):
                    if item_setx[i] != item_sety[i]:
                        flag = False
                        break
                if flag:
                    nextGen.add(item_setx + (item_sety[-1],))
        nextGen = Apriori.has_infrequent_subset(nextGen, set(freq_set))
        return sorted(nextGen)

    @staticmethod
    def has_infrequent_subset(candidates, frequent):
        res = []
        for candidate in candidates:
            flag = True
            for i in range(len(candidate)):
                if candidate[:i] + candidate[i + 1:] not in frequent:
                    flag = False
                    break
            if flag:
                res.append(candidate)
        return res


class FPGrowth(Algorithm):
    class Node:
        def __init__(self, item):
            self.item = item if item is not None else ""
            self.count = 1
            self.child = {}
            self.father = None

        def increment(self):
            self.count += 1
            return self

        def addChild(self, node):
            if node.item not in self.child:
                self.child[node.item] = node
                self.child[node.item].setFather(self)
            node.increment()
            return node

        def setFather(self, node):
            self.father = node

        def getChild(self, item):
            if item in self.child:
                return self.child[item]
            else:
                return None

        def __str__(self):
            return self.item

    class Tree:
        def __init__(self, dataset, support, confidence):
            self.itemsMapper = defaultdict(lambda: [])
            self.count = len(dataset)
            self.support = support
            self.confidence = confidence
            self.root = self.constructTree(dataset)

        def constructTree(self, dataset):
            root = FPGrowth.Node(None)
            counter = defaultdict(lambda: 0)
            for data in dataset:
                for item in data:
                    counter[item] += 1
            counter = {key: val for key, val in counter.items() if val > self.support * self.count}
            candidates = set(counter.keys())
            last_node = root
            for data in dataset:
                data = sorted(data)
                for item in data:
                    if item in candidates:
                        node = last_node.getChild(item)
                        if node is None:
                            node = FPGrowth.Node(item)
                            self.itemsMapper[item].append(node)
                        last_node = last_node.addChild(node)
                last_node = root
            return root

    def run(self, dataset):
        return FPGrowth.Tree(dataset, self.support, self.confidence)


class AssociationRuleMiner:
    def __init__(self, support, confidence, count):
        self.support = support
        self.confidence = confidence
        self.count = count

    def generateSubset(self, candidates, depth=0, record=None):
        if record is None:
            record = []
        if depth == len(candidates):
            if len(record) != 0 and len(record) != len(candidates):
                subset = tuple(sorted(list(record)))
                oppo_subset = tuple(sorted(list(set(candidates).difference(set(record)))))
                # if subset in self.itemsetCount and oppo_subset in self.itemsetCount:
                yield subset, oppo_subset
            return
        for _ in self.generateSubset(candidates, depth + 1, record=record + [candidates[depth]]):
            yield _
        for _ in self.generateSubset(candidates, depth + 1, record=record):
            yield _

    @timer()
    def association_rules(self, k_candidates, itemsetCount):
        rules = []
        for k_candidate in k_candidates:
            candidate_count = itemsetCount[k_candidate]
            for subset, opposite_set in self.generateSubset(k_candidate):
                confidence = candidate_count / itemsetCount[subset]
                known_freq = itemsetCount[opposite_set] / self.count
                if confidence > self.confidence and confidence / known_freq > 1:
                    rules.append([subset, opposite_set, confidence])
        rules = sorted(rules, key=lambda x: x[2], reverse=True)
        for rule in rules[:2000]:
            print(",".join(rule[0]), " -> ", ",".join(rule[1]), rule[2])
        return rules


if __name__ == '__main__':
    # Dummy(0.01, 0.3).run(GroceryReader().read())
    Apriori(0.01, 0.3).run(UnixReader().read())
    # FPGrowth(0.01, 0.3).run(GroceryReader().read())
    # for item in Apriori.generateSubset([1, 2, 3, 4]):
    #     print(item)
