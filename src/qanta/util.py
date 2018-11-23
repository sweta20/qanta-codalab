import click
import subprocess
from os import path, makedirs
import logging

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
