import click
import subprocess
from os import path, makedirs
import logging
import abc
from collections import defaultdict
from urllib import request
import numpy as np
import torch
from torch.autograd import Variable
import logging
import tempfile
from typing import Tuple, List, Any, Dict, Optional
import abc

DS_VERSION = '2018.04.18'
S3_HTTP_PREFIX = 'https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/'
QANTA_MAPPED_DATASET_PATH = f'qanta.mapped.{DS_VERSION}.json'
QANTA_TRAIN_DATASET_PATH = f'qanta.train.{DS_VERSION}.json'
QANTA_DEV_DATASET_PATH = f'qanta.dev.{DS_VERSION}.json'
QANTA_TEST_DATASET_PATH = f'qanta.test.{DS_VERSION}.json'
WIKI_FILE_PATH = 'wiki_lookup.json'
s3_wiki = 'https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/datasets/wikipedia/wiki_lookup.json'


FILES = [
    QANTA_MAPPED_DATASET_PATH,
    QANTA_TRAIN_DATASET_PATH,
    QANTA_DEV_DATASET_PATH,
    QANTA_TEST_DATASET_PATH
]

QuestionText = str
Page = str
Evidence = Dict[str, Any]
TrainingData = Tuple[List[List[QuestionText]], List[Page], Optional[List[Evidence]]]


def create_save_model(model):
    def save_model(path):
        torch.save(model, path)
    return save_model


def safe_path(path_name):
    makedirs(path.dirname(path_name), exist_ok=True)
    return path_name

def get(name, file_name="qanta.log"):
    log = logging.getLogger(name)

    if len(log.handlers) < 2:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        fh = logging.FileHandler(file_name)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)

        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(formatter)

        log.addHandler(fh)
        log.addHandler(sh)
        log.setLevel(logging.INFO)
    return log

def make_file_pairs(file_list, source_prefix, target_prefix):
    return [(path.join(source_prefix, f), path.join(target_prefix, f)) for f in file_list]


def shell(command):
    return subprocess.run(command, check=True, shell=True, stderr=subprocess.STDOUT)


def download_file(http_location, local_location):
    print(f'Downloading {http_location} to {local_location}')
    makedirs(path.dirname(local_location), exist_ok=True)
    shell(f'wget -O {local_location} {http_location}')


def download(local_qanta_prefix):
    """
    Download the qanta dataset
    """
    for s3_file, local_file in make_file_pairs(FILES, S3_HTTP_PREFIX, local_qanta_prefix):
        download_file(s3_file, local_file)

    download_file(s3_wiki, path.join(local_qanta_prefix, WIKI_FILE_PATH))



class Callback(abc.ABC):
    @abc.abstractmethod
    def on_epoch_end(self, logs) -> Tuple[bool, Optional[str]]:
        pass


class BaseLogger(Callback):
    def __init__(self, log_func=print):
        self.log_func = log_func
    def on_epoch_end(self, logs):
        msg = 'Epoch {}: train_acc={:.4f} test_acc={:.4f} | train_loss={:.4f} test_loss={:.4f} | time={:.1f}'.format(
            len(logs['train_acc']),
            logs['train_acc'][-1], logs['test_acc'][-1],
            logs['train_loss'][-1], logs['test_loss'][-1],
            logs['train_time'][-1]
        )
        self.log_func(msg)

    def __repr__(self):
        return 'BaseLogger()'


class TerminateOnNaN(Callback):
    def on_epoch_end(self, logs):
        for _, arr in logs.items():
            if np.any(np.isnan(arr)):
                raise ValueError('NaN encountered')
        else:
            return False, None

    def __repr__(self):
        return 'TerminateOnNaN()'


class EarlyStopping(Callback):
    def __init__(self, monitor='test_loss', min_delta=0, patience=1, verbose=0, log_func=print):
        if monitor.endswith('loss'):
            self.improvement_sign = 1
        elif monitor.endswith('acc'):
            self.improvement_sign = -1
        else:
            raise ValueError('Unrecognized monitor')
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.best_monitor_score = self.improvement_sign * float('inf')
        self.current_patience = patience
        self.verbose = verbose
        self.log_func = log_func

    def __repr__(self):
        return 'EarlyStopping(monitor={}, min_delta={}, patience={})'.format(
            self.monitor, self.min_delta, self.patience)

    def on_epoch_end(self, logs):
        if logs[self.monitor][-1] * self.improvement_sign < self.improvement_sign * self.best_monitor_score:
            self.current_patience = self.patience
            self.best_monitor_score = logs[self.monitor][-1]
        else:
            self.current_patience -= 1
            if self.verbose > 0:
                self.log_func('Patience: reduced by one and waiting for {} epochs for improvement before stopping'.format(self.current_patience))

        if self.current_patience == 0:
            return True, 'Ran out of patience'
        else:
            return False, None


class MaxEpochStopping(Callback):
    def __init__(self, max_epochs):
        self.max_epochs = max_epochs

    def on_epoch_end(self, logs):
        if len(logs['train_time']) == self.max_epochs:
            return True, 'Max epochs reached'
        else:
            return False, None


class ModelCheckpoint(Callback):
    def __init__(self, save_function, filepath, monitor='test_loss', save_best_only=True, verbose=0, log_func=print):
        self.save_function = save_function
        self.filepath = filepath
        self.save_best_only = save_best_only
        if monitor.endswith('loss'):
            self.improvement_sign = 1
        elif monitor.endswith('acc'):
            self.improvement_sign = -1
        else:
            raise ValueError('Unrecognized monitor')
        self.monitor = monitor
        self.best_monitor_score = self.improvement_sign * float('inf')
        self.verbose = verbose
        self.log_func = log_func

    def on_epoch_end(self, logs):
        if logs[self.monitor][-1] * self.improvement_sign < self.improvement_sign * self.best_monitor_score:
            self.best_monitor_score = logs[self.monitor][-1]
            if self.save_best_only:
                if self.verbose > 0:
                    self.log_func('New best model, saving to: {}'.format(self.filepath))
                self.save_function(self.filepath)
            else:
                path = self.filepath.format(epoch=len(logs['train_time']) - 1)
                if self.verbose > 0:
                    self.log_func('New best model, saving to: {}'.format(path))
                self.save_function(path)


class TrainingManager:
    def __init__(self, callbacks: List[Callback]):
        self.callbacks = callbacks
        self.logs = defaultdict(list)

    def instruct(self, train_time, train_loss, train_acc, test_time, test_loss, test_acc):
        self.logs['train_time'].append(train_time)
        self.logs['train_loss'].append(train_loss)
        self.logs['train_acc'].append(train_acc)
        self.logs['test_time'].append(test_time)
        self.logs['test_loss'].append(test_loss)
        self.logs['test_acc'].append(test_acc)

        callback_stop_reasons = []
        for c in self.callbacks:
            result = c.on_epoch_end(self.logs)
            if result is None:
                stop_training, reason = False, None
            else:
                stop_training, reason = result
            if stop_training:
                callback_stop_reasons.append('{}: {}'.format(c.__class__.__name__, reason))

        if len(callback_stop_reasons) > 0:
            return True, callback_stop_reasons
        else:
            return False, []
