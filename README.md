# Deep Batch Active Learning for Regression
[![Identifier](https://img.shields.io/badge/doi-10.18419%2Fdarus--807-d45815.svg)](https://doi.org/10.18419/darus-3110)

This repository contains code accompanying our paper ["A Framework and Benchmark for Deep Batch Active Learning for Regression"](https://arxiv.org/abs/2203.09410). It can be used for the following purposes:
- Apply various pool-based Batch Mode Deep Active Learning (BMDAL) algorithms for regression to custom neural networks (NNs) or kernel methods
- Use our NN for tabular regression through a simple scikit-learn style interface
- Download large tabular regression data sets from our benchmark
- Compare BMDAL algorithms using our benchmark

If you use this code for research purposes, plese cite [our paper](https://arxiv.org/abs/2203.09410).

## Versions

- The commit corresponding to [version 1](https://arxiv.org/abs/2203.09410v1) of our arXiv paper is tagged `arxiv_v1` and also archived with the corresponding data at [DaRUS](https://doi.org/10.18419/darus-2615).
- The commit corresponding to [version 2](https://arxiv.org/abs/2203.09410v2) of our arXiv paper is tagged `arxiv_v2` and also archived with the corresponding data at [DaRUS](https://doi.org/10.18419/darus-3110).
Results from versions 1 and 2 are run with slightly different options, hence they should not be mixed though the numbers (except for the runtimes) are very similar. Changes in version 2 are listed below.

## License

This source code is licensed under the Apache 2.0 license. However, the implementation of the acs-rf-hyper kernel transformation in `bmdal/features.py` is adapted from the source code at [https://github.com/rpinsler/active-bayesian-coresets](https://github.com/rpinsler/active-bayesian-coresets), which comes with its own (non-commercial) license. Please respect this license when using the acs-rf-hyper transformation directly from `bmdal/features.py` or indirectly through the interface provided at `bmdal/algorithms.py`.

## Installation

This code has been tested with Python 3.9.2 but may be compatible with versions down to Python 3.6. 

### Through pip
For running our NN and the active learning methods, a `pip` installation is sufficient. The library can be installed via 
```
pip3 install bmdal_reg
```
When using our benchmarking code through a `pip` installation, the paths where experiment data and plots are saved can be modified through changing the corresponding path variables of `bmdal_reg.custom_paths.CustomPaths` before running the benchmark.

### Manually

For certain purposes, especially trying new methods and running the benchmark, it might be helpful or necessary to modify the code. For this, the code can be manually installed via cloning the [GitHub repository](https://github.com/dholzmueller/bmdal_reg) and then following the instructions below:

The following packages (available through `pip`) need to be installed:
- General: `torch`, `numpy`, `dill`
- For running experiments with `run_experiments.py`: `psutil`
- For plotting the experiment results: `matplotlib`, `seaborn`
- For downloading the data sets with `download_data.py`: `pandas`, `openml`, `mat4py`

If you want to install PyTorch with GPU support, please follow the instructions [on the PyTorch website](https://pytorch.org/get-started/locally/). The following command installs the versions of the libraries we used for running the benchmark:
```
pip3 install -r requirements.txt
```
Alternatively, the following command installs current versions of the packages:
```
pip3 install torch numpy dill psutil matplotlib seaborn pandas openml mat4py
```

Clone the repository (or download the files from the repository) and change to its folder:
```
git clone git@github.com:dholzmueller/bmdal_reg.git
cd bmdal_reg
```
Then, copy the file `custom_paths.py.default` to `custom_paths.py` via
```
cp custom_paths.py.default custom_paths.py
```
and, if you want to, adjust the paths in `custom_paths.py` to specify the folders in which you want to save data and results.

## Downloading data

If you want to use the benchmark data sets, you need to download and preprocess them. We do not provide preprocessed versions of the data sets to avoid copyright issues, but you can download and preprocess the data sets using
```
python3 download_data.py
```
Note that this may take a while. This depends of course on your download speed. The preprocessing is mostly fast, but for the (large) methane data set, it took around five minutes and 25 GB of RAM for us. If you cannot download/process the data due to limited RAM, please contact the main developer (see below).

## Usage

Depending on your use case, some of the following introductory Jupyter notebooks may be helpful:
- [examples/benchmark.ipynb](https://github.com/dholzmueller/bmdal_reg/blob/main/examples/benchmark.ipynb) shows how to download or reproduce our experimental results, how to benchmark other methods, and how to evaluate the results.
- [examples/using_bmdal.ipynb](https://github.com/dholzmueller/bmdal_reg/blob/main/examples/using_bmdal.ipynb) shows how to apply our BMDAL framework to your use-case.
- [examples/framework_details.ipynb](https://github.com/dholzmueller/bmdal_reg/blob/main/examples/framework_details.ipynb) explains how our BMDAL framework is implemented, which may be relevant for advanced usage.
- *New:* [examples/nn_interface.ipynb](https://github.com/dholzmueller/bmdal_reg/blob/main/examples/nn_interface.ipynb) shows how our NN configuration can be used (without active learning) through a simple scikit-learn style interface.

Besides these notebooks, you can also take a look at the code directly. The more important parts of our code are documented with docstrings.

## Code structure

The code is structured as follows:
- The `bmdal` folder contains the implementation of all BMDAL methods, with its main interface in `bmdal/algorithms.py`.
- The `evaluation` folder contains code for analyzing and plotting generated data, which is called from `run_evaluation.py`.
- The `examples` folder contains Jupyter Notebooks for instructive purposes as mentioned above.
- The file `download_data.py` allows for downloading the data, `run_experiments.py` allows for starting the experiments, `test_single_task.py` allows for testing a configuration on a data set, and `rename_algs.py` contains some functionality for adjusting experiment data in case of mistakes. 
- The file `check_task_learnability.py` has been used to check the reduction in RMSE on different data sets when going from 256 to 4352 random samples. We used this to sort out the data sets where the reduction in RMSE was too small, since these data sets are unlikely to make a substantial difference in the benchmark results.
- The files `data.py`, `layers.py`, `models.py`, `task_execution.py`, `train.py` and `utils.py` implement parts of data loading, training, and parallel execution.

## Updates to the second version of the benchmark

- Added the BAIT selection method with variants BAIT-F and BAIT-FB.
- For the normalization of input data, mean and standard deviations for the features are now computed on training and pool set instead of only on the initial training set.
- More precise runtime measurement through CUDA synchronize (only applied in one of the 20 splits where only one process is run per GPU).
- Now, 64-bit floating point computations are used for computations involving posterior transformations. This can sometimes cause RAM overflows when using parallel execution, though. 
- We use $\sigma^2 = 10^{-6}$ instead of $\sigma^2 = 10^{-4}$ now, which still works well due to the change to 64-bit floats.
- The computation of the last-layer kernel does not require the full backward pass now since the earlier layers set `requires_grad=False` for the computation.
- Fixed a discrepancy between the implementation of selection methods and the corresponding paper pseudocode: Previously, some selection methods could re-select already selected samples in case of numerical issues, which triggered a code filling up the batch with random samples. Now, selecting already selected samples is explicitly prevented.
- Changed the interface of `run_experiments.py` to be based on lists instead of callbacks.

## Contributors

- [David Holzmüller](https://www.isa.uni-stuttgart.de/en/institute/team/Holzmueller/) (main developer)
- [Viktor Zaverkin](https://www.itheoc.uni-stuttgart.de/institute/team/Zaverkin/) (contributed to the evaluation code)

If you would like to contribute to the code or would be interested in additional features, please contact David Holzmüller.








