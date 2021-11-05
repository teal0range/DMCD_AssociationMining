from abc import abstractmethod
import pandas as pd
import json
import pathlib
import regex as re
from tqdm import tqdm


class FormatReader:
    def __init__(self):
        super().__init__()

    @abstractmethod
    def read(self):
        pass


class GroceryReader(FormatReader):
    def read(self):
        df = pd.read_csv("dataset/GroceryStore/Groceries.csv", index_col=0)
        df['items'] = df['items'].apply(lambda x: set(x[1:-1].split(",")))
        return [sorted(list(items)) for items in df['items'].tolist()]


class UnixReader(FormatReader):
    def read(self):
        result = []
        for path in tqdm(pathlib.Path("dataset/UNIX_usage").rglob("*.[0-9]*"), total=9,
                         desc='loading dataset UNIX_usage'):
            with open(path) as fp:
                line = fp.readline()
                container = set()
                while line != "":
                    if line == "**SOF**":
                        container.clear()
                    elif line == "**EOF**":
                        if len(container) != 0:
                            result.append(container.copy())
                    else:
                        container.add(line)
                    line = fp.readline().strip()
        return [sorted(list(items)) for items in result]


if __name__ == '__main__':
    print(UnixReader().read())
