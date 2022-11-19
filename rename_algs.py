from pathlib import Path
import shutil
from typing import *

from . import custom_paths
from . import utils

# This file contains some utility functions to modify/rename/remove saved results.
# It can be used for example if the names of some experiment results should be changed.


def rename_alg(exp_name: str, old_name: str, new_name: str):
    print(f'Renaming alg "{old_name}" to "{new_name}" for {exp_name} experiments')
    results_path = Path(custom_paths.get_results_path()) / exp_name
    for task_path in results_path.iterdir():
        if utils.existsDir(task_path / old_name):
            shutil.move(task_path / old_name, task_path / new_name)


def remove_alg(exp_name: str, alg_name: str):
    print(f'Removing alg "{alg_name}" for {exp_name} experiments')
    results_path = Path(custom_paths.get_results_path()) / exp_name
    for task_path in results_path.iterdir():
        if utils.existsDir(task_path / alg_name):
            shutil.rmtree(task_path / alg_name)


def replace_in_alg_name(exp_name: str, old_name: str, new_name: str):
    print(f'Replacing "{old_name}" with "{new_name}" in alg names for {exp_name} experiments')
    results_path = Path(custom_paths.get_results_path()) / exp_name
    for task_path in results_path.iterdir():
        for alg_path in task_path.iterdir():
            alg_name = str(alg_path.name)
            new_alg_name = alg_name.replace(old_name, new_name)
            if alg_name != new_alg_name:
                shutil.move(task_path / alg_name, task_path / new_alg_name)


def process_results(exp_name: str, f: Callable):
    print('Applying function to results for {exp_name} experiments')
    results_path = Path(custom_paths.get_results_path()) / exp_name
    for task_path in results_path.iterdir():
        for alg_path in task_path.iterdir():
            for split_path in alg_path.iterdir():
                file_path = split_path / 'results.json'
                if utils.existsFile(file_path):
                    results = utils.deserialize(file_path, use_json=True)
                    results = f(results)
                    utils.serialize(file_path, results, use_json=True)


if __name__ == '__main__':
    pass


