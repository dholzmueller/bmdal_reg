import os
import os.path
import heapq
import glob
import gzip
import shutil
import copy
import timeit
import time
import numpy as np
from typing import *
import dill
import json
import itertools


def select_from_config(config, keys):
    selected = {}
    for key in keys:
        if key in config:
            selected[key] = config[key]
    return selected


def adapt_config(config, **kwargs):
    new_config = copy.deepcopy(config)
    for key, value in kwargs.items():
        new_config[key] = value
    return new_config


def existsDir(directory):
    if directory != '':
        if not os.path.exists(directory):
            return False
    return True


def existsFile(file_path):
    return os.path.isfile(file_path)


def ensureDir(file_path):
    directory = os.path.dirname(file_path)
    if directory != '':
        if not os.path.exists(directory):
            os.makedirs(directory)


def matchFiles(file_matcher):
    return glob.glob(file_matcher)


def newDirname(prefix):
    i = 0
    name = prefix
    if existsDir(prefix):
        while existsDir(prefix + "_" + str(i)):
            i += 1
        name = prefix + "_" + str(i)
    os.makedirs(name)
    return name


def getSubfolderNames(folder):
    return [os.path.basename(name)
            for name in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, name))]


def getSubfolders(folder):
    return [os.path.join(folder, name)
            for name in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, name))]


def writeToFile(filename, content):
    ensureDir(filename)
    file = open(filename, 'w')
    file.truncate()
    file.write(content)
    file.close()


def readFromFile(filename):
    if not os.path.isfile(filename):
        return ''

    file = open(filename, 'r')
    result = file.read()
    file.close()
    return result


def serialize(filename, obj, compressed=False, use_json=False):
    # json only works for nested dicts
    ensureDir(filename)
    if compressed:
        file = gzip.open(filename, 'w' if use_json else 'wb')
    else:
        file = open(filename, 'w' if use_json else 'wb')
    # dill can dump lambdas, and dill also dumps the class and not only the contents
    if use_json:
        json.dump(obj, file)
    else:
        dill.dump(obj, file)
    file.close()


def deserialize(filename, compressed=False, use_json=False):
    # json only works for nested dicts
    if compressed:
        file = gzip.open(filename, 'r' if use_json else 'rb')
    else:
        file = open(filename, 'r' if use_json else 'rb')
    if use_json:
        result = json.load(file)
    else:
        result = dill.load(file)
    file.close()
    return result


def copyFile(src, dst):
    ensureDir(dst)
    shutil.copyfile(src, dst)


def nsmallest(n, inputList):
    return heapq.nsmallest(n, inputList)[-1]


def identity(x):
    return x


def set_none_except(lst, idxs):
    for i in range(len(lst)):
        if i not in idxs:
            lst[i] = None


def argsort(lst, key: Optional[Callable] = None):
    # from https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
    if key is None:
        range_key = lst.__getitem__
    else:
        range_key = lambda i, f=key: f(lst[i])
    return sorted(range(len(lst)), key=range_key)


def dict_argsort(dict_to_sort: dict, key: Optional[Callable] = None):
    keys = list(dict_to_sort.keys())
    values = list(dict_to_sort.values())
    perm = argsort(values, key=key)
    return [keys[i] for i in perm]


def join_dicts(*dicts):
    # Attention: arguments do not commute since later dicts can override entries from earlier dicts!
    result = copy.copy(dicts[0])
    for d in dicts[1:]:
        result.update(d)
    return result


def update_dict(d: dict, update: Optional[dict] = None, remove_keys: Optional[Union[object, List[object]]] = None):
    d = copy.copy(d)
    if update is not None:
        d.update(update)
    if remove_keys is not None:
        if isinstance(remove_keys, List):
            for key in remove_keys:
                d.pop(key)
        else:
            d.pop(remove_keys)
    return d


def pretty_table_str(str_table):
    max_lens = [np.max([len(row[i]) for row in str_table])for i in range(len(str_table[0]))]
    whole_str = ''
    for row in str_table:
        for i, entry in enumerate(row):
            whole_str += entry + (' ' * (max_lens[i] - len(entry)))
        whole_str += '\n'
    return whole_str[:-1]  # remove last newline


def prod(it: Iterable, id=None):
    result = None
    for value in it:
        if result is None:
            result = value
        else:
            result = result * value
    if result is None:
        if id is None:
            raise ValueError(f'Cannot compute empty product without identity element')
        else:
            return id
    return result


def all_equal(it: Iterable):
    # see https://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-identical
    g = itertools.groupby(it)  # iterates over unique elements
    try:
        next(g)
        next(g)
    except StopIteration:
        return True
    return False


class Timer:
    def __init__(self):
        self.start_time_total = None
        self.start_time_process = None
        self.acc_time_total = 0.0
        self.acc_time_process = 0.0

    def start(self):
        if self.start_time_total is None or self.start_time_process is None:
            self.start_time_total = timeit.default_timer()
            self.start_time_process = time.process_time()

    def pause(self):
        if self.start_time_total is None or self.start_time_process is None:
            return  # has already been paused or not been started
        self.acc_time_total += timeit.default_timer() - self.start_time_total
        self.acc_time_process += time.process_time() - self.start_time_process
        self.start_time_total = None
        self.start_time_process = None

    def get_result_dict(self):
        return {'total': self.acc_time_total, 'process': self.acc_time_process}


class TimePrinter:
    def __init__(self, desc: str):
        self.desc = desc
        self.timer = Timer()

    def __enter__(self):
        self.timer.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.timer.pause()
        print(f'Time for {self.desc}: {self.timer.get_result_dict()["total"]:g}s')


def format_length_s(duration: float) -> str:
    seconds = int(duration)
    minutes = seconds // 60
    seconds -= minutes * 60
    hours = minutes // 60
    minutes -= hours * 60
    days = hours // 24
    hours -= days * 24

    result = f'{seconds}s'
    if minutes > 0:
        result = f'{minutes}m' + result
    if hours > 0:
        result = f'{hours}h' + result
    if days > 0:
        result = f'{days}d' + result

    return result


def format_date_s(time_s: float) -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time_s))


def get_batch_intervals(n_total: int, batch_size: int) -> List[Tuple[int, int]]:
    boundaries = [i * batch_size for i in range(1 + n_total // batch_size)]
    if boundaries[-1] != n_total:
        boundaries.append(n_total)
    return [(start, stop) for start, stop in zip(boundaries[:-1], boundaries[1:])]


def last_mod_time_recursive(path: str):
    # see https://stackoverflow.com/questions/29685069/get-the-last-modified-date-of-a-directory-including-subdirectories-using-pytho
    import os
    return max(os.path.getmtime(root) for root, _, _ in os.walk(path))





