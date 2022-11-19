import pandas as pd
import numpy as np
import requests
import shutil
from typing import *
import openml
import mat4py

from . import custom_paths
from . import utils
from .data import DataInfo



def download_if_not_exists(url: str, dest: str):
    """
    Simple function for downloading a file from an url if no file at the destination path exists.
    :param url: URL of the file to download.
    :param dest: Path where to save the downloaded file.
    """
    # following https://dzone.com/articles/simple-examples-of-downloading-files-using-python
    utils.ensureDir(dest)
    if not utils.existsFile(dest):
        print('Downloading ' + url, flush=True)
        # file = requests.get(url)
        # open(dest, 'wb').write(file.content)
        r = requests.get(url, stream=True)
        with open(dest, 'wb') as f:
            print('Progress (dot = 1 MB): ', end='', flush=True)
            for ch in r.iter_content(chunk_size=1024**2):
                print('.', end='', flush=True)
                f.write(ch)
            print(flush=True)


class PandasTask:
    """
    This class represents a task (data set with indicated target variable) given by Pandas DataFrames.
    Additionally, a dedicated train-test split can be specified
    and the name of the data set needs to be specified for saving.
    This class provides a variety of methods for altering the task by different preprocessing methods.
    """
    def __init__(self, x_df: pd.DataFrame, y_df: pd.Series, ds_name: str, cat_indicator: Optional[List[bool]] = None,
                 train_test_split: Optional[int] = None):
        """
        :param x_df: DataFrame containing the inputs (covariates).
        :param y_df: pd.Series containing the targets.
        :param ds_name: Name for saving the data set.
        :param cat_indicator: Optional.
        One may specify a list of booleans which indicate whether each column of x is a category (True) or not (False).
        Otherwise, the column types in x_df will be used to decide whether a column is categorical or not.
        :param train_test_split: Optional. An integer can be specified as the index of the first test sample,
        if the data set has a dedicated test set part at the end.
        """
        if cat_indicator is None:
            cat_indicator = [not pd.api.types.is_numeric_dtype(x_df[x_df.columns[i]]) for i in range(len(x_df.columns))]
        else:
            # this is a fix coming from a different codebase
            # because category_indicator[0] was False for the dataset MIP-2016-regression
            # despite the column being categorical  (dtype=object)
            cat_indicator = [v or not pd.api.types.is_numeric_dtype(x_df[x_df.columns[i]])
                             for i, v in enumerate(cat_indicator)]
        if len(x_df.columns) != len(cat_indicator):
            raise ValueError('x.shape[1] != len(cat_indicator)')

        self.x_df = x_df  # should be (sparse) pd.DataFrame
        # should be (sparse) pd.Series  (i.e. a single column of a DataFrame)
        self.y_df = y_df
        self.ds_name = ds_name

        self.cat_cols = [x_df.columns[i] for i in range(len(x_df.columns)) if cat_indicator[i]]
        self.cont_cols = [x_df.columns[i] for i in range(len(x_df.columns)) if not cat_indicator[i]]
        self.train_test_split = train_test_split

    def get_n_samples(self):
        """
        :return: Returns the number of samples (number of rows in the DataFrame).
        """
        return len(self.x_df)

    def remove_missing_cont(self):
        """
        Removes rows with missing values in continuous columns.
        """
        print('removing columns with missing continuous values')
        if len(self.cont_cols) == 0:
            return  # no continuous columns

        not_nan_rows = self.x_df.notna().all(axis=1)
        self.x_df = self.x_df.loc[not_nan_rows, :]
        self.y_df = self.y_df.loc[not_nan_rows]

    def normalize_regression_y(self):
        """
        Centers and standardizes the target variable.
        """
        print('normalizing regression y')
        y_np = np.asarray(self.y_df)
        self.y_df.loc[:] = (y_np - np.mean(y_np)) / (np.std(y_np) + 1e-30)

    def subsample_dfs_(self, dfs: List[pd.DataFrame], max_n_samples: int) -> List[pd.DataFrame]:
        """
        Internal method for jointly subsampling multiple Pandas DataFrames of the same number of rows.
        :param dfs: Data Frames.
        :param max_n_samples: Maximum number of remaining rows.
        :return: Returns a List of potentially subsampled Pandas DataFrames.
        """
        if len(dfs[0]) <= max_n_samples:
            return dfs
        print(f'subsampling from {len(dfs[0])} samples to {max_n_samples}')
        idxs = np.random.default_rng(12345).permutation(len(dfs[0]))[:max_n_samples]
        return [df.iloc[idxs] for df in dfs]

    def subsample(self, max_tvp_samples: int, max_test_samples: int):
        """
        Subsamples the data set if necessary to not exceed a given maximum size.
        :param max_tvp_samples: Maximum number of train+val+pool samples.
        :param max_test_samples: Maximum number of test samples.
        """
        if self.train_test_split is not None:
            dfs_train = self.subsample_dfs_([self.x_df.loc[:self.train_test_split],
                                             self.y_df.loc[:self.train_test_split]], max_n_samples=max_tvp_samples)
            dfs_test = self.subsample_dfs_([self.x_df.loc[self.train_test_split:],
                                            self.y_df.loc[self.train_test_split:]], max_n_samples=max_test_samples)
            self.train_test_split = len(dfs_train[0])
            self.x_df = pd.concat([dfs_train[0], dfs_test[0]], axis=0)
            self.y_df = pd.concat([dfs_train[1], dfs_test[1]], axis=0)
        else:
            dfs = self.subsample_dfs_([self.x_df, self.y_df], max_n_samples=max_tvp_samples + max_test_samples)
            self.x_df, self.y_df = dfs[0], dfs[1]

    def remove_constant_columns(self):
        """
        Removes columns with only a single value (this could happen after removing NaN values).
        """
        # taken from https://stackoverflow.com/questions/20209600/pandas-dataframe-remove-constant-column
        non_constant_columns = (self.x_df != self.x_df.iloc[0]).any()
        print(f'removing constant columns')
        self.x_df = self.x_df.loc[:, non_constant_columns]
        self.cat_cols = [key for key in self.cat_cols if key in self.x_df.columns]
        self.cont_cols = [key for key in self.cont_cols if key in self.x_df.columns]

    def one_hot_encode(self, max_one_hot_columns: int):
        """
        Applies one-hot encoding to categorical columns.
        :param max_one_hot_columns: Maximal number of allowed one-hot encoded columns.
        If more one-hot encoded columns would be generated,
        the categorical columns with the largest number of categories are not one-hot encoded.
        """
        cat_cols_with_size = [(col_name, self.x_df.loc[:, col_name].nunique())
                              for i, col_name in enumerate(self.cat_cols)]
        if len(cat_cols_with_size) == 0:
            return  # nothing to encode
        print('one-hot encoding columns')
        cat_cols_with_size.sort(key=lambda t: t[1])  # sort by size of category
        max_cat_size = cat_cols_with_size[-1][1]
        new_col_sum = 0
        for key, sz in cat_cols_with_size:
            new_col_sum += sz
            if new_col_sum > max_one_hot_columns:
                max_cat_size = sz-1
                break

        new_cat_cols = []

        for key, sz in cat_cols_with_size:
            if sz <= max_cat_size:
                print(f'one-hot encoding column {key} with {sz} unique elements')
                # following https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python
                # https://queirozf.com/entries/one-hot-encoding-a-feature-on-a-pandas-dataframe-an-example
                col = self.x_df[key].astype('category')
                dummies = pd.get_dummies(col, prefix=f'{key}_onehot_', drop_first=(sz == 2), dummy_na=True,
                                         dtype=np.int32)
                self.cont_cols.extend(list(dummies.columns))
                self.x_df.drop([key], axis=1, inplace=True)
                self.x_df = pd.concat([self.x_df, dummies], axis=1)
            else:
                new_cat_cols.append(key)
                print(f'categorical column {key} with {sz} unique values is not one-hot encoded due to size constraints')

        self.cat_cols = new_cat_cols

    def save(self, n_test: int):
        """
        Saves the data set in the folder f'{custom_paths.get_data_path()}/data/{self.ds_name}'.
        :param n_test: Desired number of test samples.
        """
        folder = f'{custom_paths.get_data_path()}/data/{self.ds_name}'
        x_cont = np.array(self.x_df.reindex(columns=self.cont_cols), dtype=np.float32)
        y = np.array(self.y_df, dtype=np.float32)
        n_tvp = self.get_n_samples() - n_test
        data_info = DataInfo(ds_name=self.ds_name, n_tvp=n_tvp, n_test=n_test,
                             n_features=x_cont.shape[1],
                             train_test_split=self.train_test_split)

        utils.serialize(f'{folder}/data_info.pkl', data_info)
        np.save(f'{folder}/X.npy', x_cont)
        np.save(f'{folder}/y.npy', y[:, None])

        # ds = DictDataset({'x_cont': torch.as_tensor(x_cont), 'x_cat': torch.as_tensor(x_cat),
        #                   'y': torch.as_tensor(y[:, None])},
        #                  {'x_cont': TensorInfo(feat_shape=[x_cont.shape[-1]]),
        #                   'x_cat': TensorInfo(cat_sizes=cat_sizes),
        #                   'y': TensorInfo(cat_sizes=[self.get_n_classes()])})
        # task_info = TaskInfo.from_ds(task_desc, ds)
        # return Task(task_info, ds)

    @staticmethod
    def from_openml_task_id(task_id: int, ds_name: str):
        """
        Creates a PandasTask from an OpenML task.
        If more options are desired, we refer to PandasTask.from_openml_dataset_id() instead.
        :param task_id: OpenML task id.
        :param ds_name: Short name of the data set for saving.
        :return: Returns a PandasTask representing the OpenML task.
        """
        task = openml.tasks.get_task(task_id, download_data=False)
        return PandasTask.from_openml_dataset_id(task.dataset_id, ds_name, task.target_name)

    @staticmethod
    def from_openml_dataset_id(dataset_id: int, ds_name: str, target: str,
                               ignore_columns: Optional[List[str]] = None,
                               use_log_target: bool = False):
        """
        Creates a PandasTask from an OpenML data set.
        :param dataset_id: OpenML data set id.
        :param ds_name: Short name of the data set for saving.
        :param target: Name of the target variable.
        :param ignore_columns: Optional. List of columns that should be removed.
        :param use_log_target: Whether the logarithm should be applied to the target column.
        :return: Returns a PandasTask representing the corresponding OpenML data set.
        """
        print(f'Importing dataset {ds_name}')
        openml.config.set_cache_directory('./openml_cache')
        dataset = openml.datasets.get_dataset(dataset_id, download_data=False)
        print(f'dataset name: {dataset.name}')
        # print(dataset.get_data(dataset_format='dataframe'))
        x_df, y_df, cat_indicator, names = dataset.get_data(target=target,
                                                            dataset_format='dataframe')

        if ignore_columns is not None:
            cat_indicator = [value for col_name, value in zip(x_df.columns, cat_indicator)
                                  if col_name not in ignore_columns]

        for key in ignore_columns or []:
            x_df.drop([key], axis=1, inplace=True)

        if use_log_target:
            y_df = np.log(y_df)

        print('Imported x_df:\n', x_df)
        print('Imported y_df:\n', y_df)

        return PandasTask(x_df, y_df, ds_name, cat_indicator)

    @staticmethod
    def from_uci(url: str, ds_name: str, zip_name: str, csv_name: str, target_col_idxs: List[int],
                 ignore_col_idxs: Optional[List[int]] = None, use_log_target: bool = False,
                 train_test_boundary: Optional[int] = None, has_header: bool = True,
                 continuous_nan_columns: Optional[List[int]] = None,
                 convert_to_cat_columns: Optional[List[int]] = None,
                 ignore_other_csv_files: bool = False, separator: Optional[str] = None):
        """
        Create a PandasTask object from a data set on the UCI repository.
        :param url: URL of the data set file.
        :param ds_name: Short name of the data set used for saving the data set.
        :param zip_name: Target name of the downloaded file.
        If the downloaded file is not zip/compressed, i.e., no unzipping is needed,
        zip_name should be the same as csv_name.
        Otherwise the file with name zip_name will be unzipped to the file with name csv_name.
        :param csv_name: Target name of the uncompressed file, see zip_name.
        :param target_col_idxs: List of indexes of target columns. Mostly, this will only have one element.
        :param ignore_col_idxs: List of indexes of columns to be removed.
        :param use_log_target: Whether the logarithm should be applied to the target value.
        :param train_test_boundary: Index of the first test sample.
        If (as in most cases) there is no dedicated test set, None should be specified.
        :param has_header: Whether the downloaded (unzipped) csv file has a row with column names that should be removed.
        :param continuous_nan_columns: Optional. List of indexes of numeric columns that can contain NaN values.
        This triggers a corresponding Pandas column conversion.
        :param convert_to_cat_columns: Optional.
        List of indexes of columns that should be converted to categorical type.
        :param ignore_other_csv_files: If set to True,
        unzipped csv files with file names other than csv_name will be ignored.
        If set to False, unzipped csv files with file names other than csv_name
        will be interpreted as a dedicated test set,
        and the file with name csv_name will be interpreted as the train+val+pool set.
        :param separator: Separator in the csv file. Default is ','.
        If the file with name csv_name is a tsv file, separator='\t' should be specified.
        :return: Returns a PandasTask object.
        """
        print(f'Importing dataset {ds_name}')
        base_path = custom_paths.get_data_path()
        raw_data_folder = f'{base_path}/raw_data/{ds_name}'
        zip_file = f'{raw_data_folder}/{zip_name}'
        csv_file = f'{raw_data_folder}/{csv_name}'
        download_if_not_exists(url, zip_file)
        if not utils.existsFile(csv_file):
            print('Unpacking zip file...')
            shutil.unpack_archive(zip_file, raw_data_folder)

        if separator is None:
            separator = ','

        print('Processing csv data...')
        if ignore_other_csv_files:
            non_train_files = []
        else:
            non_train_files = [file for file in utils.matchFiles(f'{raw_data_folder}/*.csv') if file != csv_file]
        df = pd.read_csv(csv_file, header='infer' if has_header else None, sep=separator)
        if len(non_train_files) > 0:
            train_test_boundary = len(df)
            df = pd.concat([df] + [pd.read_csv(file, header='infer' if has_header else None, sep=separator)
                                     for file in non_train_files])

        if continuous_nan_columns is not None:
            for col_idx in continuous_nan_columns:
                df.iloc[:, col_idx] = pd.to_numeric(df.iloc[:, col_idx], errors='coerce')

        if convert_to_cat_columns is not None:
            for col_idx in convert_to_cat_columns:
                df.iloc[:, col_idx] = df.iloc[:, col_idx].astype('category')

        input_column_names = []
        target_columns = []
        for i in range(len(df.columns)):
            if i in target_col_idxs:
                target_columns.append(df.iloc[:, i].to_numpy().astype(np.float32))
            elif ignore_col_idxs is None or i not in ignore_col_idxs:
                input_column_names.append(df.columns[i])

        y = np.median(np.stack(target_columns, axis=1), axis=1)
        if use_log_target:
            y = np.log(y)

        # https://stackoverflow.com/questions/20490274/how-to-reset-index-in-a-pandas-dataframe
        x_df = df.reindex(columns=input_column_names).reset_index(drop=True)
        y_df = pd.DataFrame({'y': y})['y']

        return PandasTask(x_df, y_df, ds_name,
                          train_test_split=train_test_boundary)


class PandasTaskPreprocessor:
    """
    This class allows to preprocess data sets given by PandasTask objects. Various options can be configured.
    """
    def __init__(self, min_n_samples: int, max_tvp_samples: int, max_test_samples: int, max_one_hot_columns: int):
        """
        :param min_n_samples: Minimum number of samples that a task must have
        after removing missing values in continuous columns.
        :param max_tvp_samples: Maximum number of samples for the train+val+pool sets.
        :param max_test_samples: Maximum number of samples for the test set.
        :param max_one_hot_columns: Maximum number of one-hot encoded columns that are allowed.
        If more would be generated, the categorical variables with the largest category sizes are removed.
        """
        self.min_n_samples = min_n_samples
        self.max_tvp_samples = max_tvp_samples
        self.max_test_samples = max_test_samples
        self.max_one_hot_columns = max_one_hot_columns

    def apply(self, pd_task: PandasTask):
        """
        Apply preprocessing to a PandasTask with the options given in the constructor
         and save the preprocessed data set under the name specified in the PandasTask.
        :param pd_task: PandasTask object holding infomation about the unprocessed data set.
        """
        pd_task.remove_missing_cont()
        if pd_task.get_n_samples() < self.min_n_samples:
            print(f'Task {pd_task.ds_name} has only {pd_task.get_n_samples()} samples after removing missing values, '
                  f'hence we discard it')
            return
        pd_task.subsample(max_tvp_samples=self.max_tvp_samples, max_test_samples=self.max_test_samples)
        pd_task.remove_constant_columns()
        pd_task.one_hot_encode(self.max_one_hot_columns)
        pd_task.normalize_regression_y()
        n_samples = pd_task.get_n_samples()
        if pd_task.train_test_split is not None:
            n_test = n_samples - pd_task.train_test_split
        else:
            n_test = max(int(0.2 * n_samples), n_samples - self.max_tvp_samples)
        pd_task.save(n_test)


def get_sarcos_pandas_task() -> PandasTask:
    """
    This is a separate function for downloading the sarcos data set, since it is not in the UCI / OpenML repositories.
    """
    print(f'Importing dataset sarcos')
    base_path = custom_paths.get_data_path()
    raw_data_folder = f'{base_path}/raw_data/sarcos'
    file_path = f'{raw_data_folder}/sarcos_inv.mat'
    download_if_not_exists('http://www.gaussianprocess.org/gpml/data/sarcos_inv.mat', file_path)
    # don't download test data since test_data = train_data[::10]
    # download_if_not_exists('http://www.gaussianprocess.org/gpml/data/sarcos_inv_test.mat',
    #                        f'{raw_data_folder}/sarcos_inv_test.mat')
    data = np.array(mat4py.loadmat(file_path)['sarcos_inv'])
    return PandasTask(x_df=pd.DataFrame(data[:, :-7]), y_df=pd.Series(data[:, -7]), ds_name='sarcos')


def import_all():
    proc = PandasTaskPreprocessor(min_n_samples=30000, max_tvp_samples=200000, max_test_samples=300000,
                                  max_one_hot_columns=300)

    proc.apply(PandasTask.from_uci('https://archive.ics.uci.edu/ml/machine-learning-databases/00440/sgemm_product_dataset.zip',
                   ds_name='sgemm', zip_name='sgemm_product_dataset.zip', csv_name='sgemm_product.csv',
                   target_col_idxs=[14, 15, 16, 17], use_log_target=True))
    proc.apply(PandasTask.from_uci('https://archive.ics.uci.edu/ml/machine-learning-databases/00206/slice_localization_data.zip',
                   ds_name='ct', zip_name='slice_localization_data.zip', csv_name='slice_localization_data.csv',
                   target_col_idxs=[385], ignore_col_idxs=[0]))
    proc.apply(PandasTask.from_uci('https://archive.ics.uci.edu/ml/machine-learning-databases/00221/Reaction%20Network%20(Undirected).data',
                   ds_name='kegg_undir_uci', zip_name='kegg_undir_uci.csv', csv_name='kegg_undir_uci.csv',
                   target_col_idxs=[26], ignore_col_idxs=[0], has_header=False, continuous_nan_columns=[4]))
    # only use the Sydney part of the data set (could as well have used Adelaide, Perth or Tasmania)
    proc.apply(PandasTask.from_uci('https://archive.ics.uci.edu/ml/machine-learning-databases/00494/WECs_DataSet.zip',
                   ds_name='wecs', zip_name='WECs_DataSet.zip', csv_name='WECs_DataSet/Sydney_Data.csv',
                   target_col_idxs=[48], has_header=False, ignore_other_csv_files=True))
    proc.apply(PandasTask.from_uci('https://archive.ics.uci.edu/ml/machine-learning-databases/00335/online_video_dataset.zip',
                                   ds_name='online_video', zip_name='online_video_dataset.zip',
                                   csv_name='transcoding_mesurment.tsv', separator='\t', ignore_col_idxs=[0, 20],
                                   target_col_idxs=[21], has_header=True, ignore_other_csv_files=True))
    proc.apply(
        PandasTask.from_uci('https://archive.ics.uci.edu/ml/machine-learning-databases/00493/datasets.zip',
                            ds_name='query_agg_count', zip_name='datasets.zip',
                            csv_name='Datasets/Range-Queries-Aggregates.csv', ignore_col_idxs=[0, 6, 7],
                            target_col_idxs=[5], has_header=True, ignore_other_csv_files=True))
    proc.apply(
        PandasTask.from_uci('https://archive.ics.uci.edu/ml/machine-learning-databases/00246/3D_spatial_network.txt',
                            ds_name='road_network', zip_name='3D_spatial_network.txt',
                            csv_name='3D_spatial_network.txt', ignore_col_idxs=[0],
                            target_col_idxs=[3], has_header=False, ignore_other_csv_files=True))
    # note: we use only the testing data here with random splits since the training data is so small
    # and the testing data so large
    proc.apply(
        PandasTask.from_uci('https://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-testing.data',
                            ds_name='poker', zip_name='data.csv', csv_name='data.csv',
                            convert_to_cat_columns=list(range(10)),
                            target_col_idxs=[10], has_header=False, ignore_other_csv_files=True))
    proc.apply(get_sarcos_pandas_task())
    proc.apply(PandasTask.from_openml_dataset_id(dataset_id=1200, ds_name='stock', target='company10'))
    proc.apply(PandasTask.from_openml_dataset_id(dataset_id=42454, ds_name='mlr_knn_rng', target='perf.logloss',
                                                 ignore_columns=['perf.mmce']))
    proc.apply(PandasTask.from_openml_dataset_id(dataset_id=42701, ds_name='methane', target='MM264',
                                                 ignore_columns=['MM263', 'MM256']))
    proc.apply(PandasTask.from_openml_dataset_id(dataset_id=42225, ds_name='diamonds', target='price'))
    proc.apply(PandasTask.from_openml_dataset_id(dataset_id=564, ds_name='fried', target='Y'))
    proc.apply(PandasTask.from_openml_dataset_id(dataset_id=42903, ds_name='protein', target='RMSD'))

    # ------ data sets that were discarded due to too many missing values or too low learnability ------
    # proc.apply(PandasTask.from_uci('https://archive.ics.uci.edu/ml/machine-learning-databases/00304/BlogFeedback.zip',
    #                ds_name='blog', zip_name='BlogFeedback.zip', csv_name='blogData_train.csv',
    #                target_col_idxs=[280], has_header=False))
    # proc.apply(PandasTask.from_uci('https://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip',
    #                ds_name='news', zip_name='OnlineNewsPopularity.zip',
    #                csv_name='OnlineNewsPopularity/OnlineNewsPopularity.csv',
    #                target_col_idxs=[60], ignore_col_idxs=[0, 1]))
    # proc.apply(PandasTask.from_uci('https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip',
    #                ds_name='year', zip_name='YearPredictionMSD.txt.zip', csv_name='YearPredictionMSD.txt',
    #                target_col_idxs=[0], train_test_boundary=463715, has_header=False))
    # # date-time currently not handled
    # proc.apply(PandasTask.from_openml_dataset_id(dataset_id=42998, ds_name='metro', target='traffic_volume'))
    # proc.apply(PandasTask.from_openml_dataset_id(dataset_id=42163, ds_name='procurement', target='award_value_euro'))
    # # ignore V1, V5 is numerical but with missing value?, V27 might not be the true target
    # # but use kegg_undir_uci anyway
    # proc.apply(PandasTask.from_openml_dataset_id(dataset_id=4533, ds_name='kegg_undir', target='V27'))
    # proc.apply(PandasTask.from_openml_dataset_id(dataset_id=41540, ds_name='black_friday', target='Purchase'))
    # proc.apply(PandasTask.from_openml_dataset_id(dataset_id=4549, ds_name='buzz_twitter', target='Annotation'))
    # proc.apply(PandasTask.from_openml_dataset_id(dataset_id=1204, ds_name='bng_wine', target='quality'))
    # proc.apply(PandasTask.from_openml_dataset_id(dataset_id=1197, ds_name='2dplanes', target='y'))
    # proc.apply(PandasTask.from_openml_dataset_id(dataset_id=1213, ds_name='mv', target='y'))
    # proc.apply(PandasTask.from_openml_dataset_id(dataset_id=42705, ds_name='yolanda', target='101'))
    # proc.apply(PandasTask.from_openml_dataset_id(dataset_id=1203, ds_name='pwlinear', target='class'))

    # ------ kegg was discarded since we already have kegg_undir_uci, which has better learnability ------
    # proc.apply(PandasTask.from_uci('https://archive.ics.uci.edu/ml/machine-learning-databases/00220/Relation%20Network%20(Directed).data',
    #                ds_name='kegg', zip_name='kegg.csv', csv_name='kegg.csv',
    #                target_col_idxs=[23], ignore_col_idxs=[0], has_header=False))


if __name__ == '__main__':
    import_all()
    pass






