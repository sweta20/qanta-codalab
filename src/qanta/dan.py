from typing import List, Optional, Tuple
from collections import defaultdict
import pickle
import json
from os import path
import numpy as np
import os
import shutil
import random
import time
import cloudpickle

import xgboost
import numpy as np

import click
from tqdm import tqdm
from flask import Flask, jsonify, request

import torch
from torch.utils.data import DataLoader
from  torch.utils.data.sampler import SequentialSampler, RandomSampler
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam, lr_scheduler
from torch.nn import functional as F

from qanta import util
from qanta.dataset import QuizBowlDataset
from qanta.preprocess import preprocess_dataset, WikipediaDataset, tokenize_question
from models import DanModel

from qanta.buzzer_utils import Buzzer

categories = {
    0: ['History', 'Philosophy', 'Religion'],
    1: ['Literature', 'Mythology'],
    2: ['Science', 'Social Science'],
    3: ['Current Events', 'Trash', 'Fine Arts', 'Geography']
}

MODEL_PATH = "./models/dan_pattern_ner"

BUZZER_PATH = MODEL_PATH + "/buzzer_"+ MODEL_PATH.split("/")[-1] + ".pkl"
BUZZ_NUM_GUESSES = 10
BUZZ_THRESHOLD = 0.2

buzzer = pickle.load(open(BUZZER_PATH, "rb"))


def guess_and_buzz(model, question_text) -> Tuple[str, bool]:
    guesses = model.guess([question_text], BUZZ_NUM_GUESSES)[0]
    scores = [guess[1] for guess in guesses]
    buzz = scores[0] / sum(scores) >= BUZZ_THRESHOLD
    return guesses[0][0], buzz

def guess_and_predict_buzz(model, question_text, char_skip=50) -> Tuple[str, bool]:
    char_indices = list(range(char_skip, len(question_text) + char_skip, char_skip))
    guesses = model.guess([question_text], BUZZ_NUM_GUESSES)[0]
    scores = [guess[1] for guess in guesses]
    buzz_preds = buzzer.predict([np.append(scores, char_indices[-1])])
    buzz = buzz_preds[0]
    return guesses[0][0], buzz

def batch_guess_and_buzz(model, questions) -> List[Tuple[str, bool]]:
    question_guesses = model.guess(questions, BUZZ_NUM_GUESSES)
    outputs = []
    for guesses in question_guesses:
        scores = [guess[1] for guess in guesses]
        buzz = scores[0] / sum(scores) >= BUZZ_THRESHOLD
        outputs.append((guesses[0][0], buzz))
    return outputs

def batch_guess_and_predict_buzz(model, questions, char_skip=50) -> List[Tuple[str, bool]]:
    question_guesses = model.guess(questions, BUZZ_NUM_GUESSES)
    outputs = []
    for i in range(len(question_guesses)):
        char_indices = list(range(char_skip, len(questions[i]) + char_skip, char_skip))
        guesses = question_guesses[i]
        scores = [guess[1] for guess in guesses]
        buzz_preds = buzzer.predict([np.append(scores, char_indices[-1])])
        buzz = buzz_preds[0]
        outputs.append((guesses[0][0], buzz))
    return outputs


def make_array(tokens, vocab, add_eos=True):
    unk_id = vocab['<unk>']
    eos_id = vocab['<eos>']
    ids = [vocab.get(token, unk_id) for token in tokens]
    if add_eos:
        ids.append(eos_id)
    return np.array(ids, 'i')

def transform_to_array(dataset, vocab, with_label=True):
    if with_label:
        return [(make_array(tokens, vocab), np.array([cls], 'i'))
                for tokens, cls in dataset]
    else:
        return [make_array(tokens, vocab)
                for tokens in dataset]

def get_quizbowl(guesser_train=True, buzzer_train=False, category=None, use_wiki=False, n_wiki_sentences = 5):
    print("Loading data with guesser_train: " + str(guesser_train) + " buzzer_train:  " + str(buzzer_train))
    qb_dataset = QuizBowlDataset(guesser_train=guesser_train, buzzer_train=buzzer_train, category=category)
    training_data = qb_dataset.training_data()
    
    if use_wiki and n_wiki_sentences > 0:
        print("Using wiki dataset with n_wiki_sentences: " + str(n_wiki_sentences))
        wiki_dataset = WikipediaDataset(set(training_data[1]), n_wiki_sentences)
        wiki_training_data = wiki_dataset.training_data()
        training_data[0].extend(wiki_training_data[0])
        training_data[1].extend(wiki_training_data[1])
    return training_data

def load_glove(filename):
    idx = 0
    word2idx = {}
    vectors = []

    with open(filename, 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)

    return word2idx, vectors


class DanGuesser:
    def __init__(self):
        self.model = None
        self.i_to_class = None
        self.class_to_i = None
        self.word_to_i = None

        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        self.model_file = None

        self.map_pattern = False
        self.wiki_links = False
        self.use_es_highlight = False
        self.full_question = False
        self.use_wiki = False

        self.name = "dan"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def batchify(self, batch):
        """
        Gather a batch of individual examples into one batch, 
        which includes the question text, question length and labels 

        Keyword arguments:
        batch: list of outputs from vectorize function
        """
        batch = transform_to_array(batch, self.word_to_i)
        question_len = list()
        label_list = list()
        for ex in batch:
            question_len.append(len(ex[0]))
            label_list.append(ex[1][0])
        target_labels = torch.LongTensor(label_list)
        x1 = torch.LongTensor(len(question_len), max(question_len)).zero_()
        for i in range(len(question_len)):
            question_text = batch[i][0]
            vec = torch.LongTensor(question_text)
            x1[i, :len(question_text)].copy_(vec)
        q_batch = {'text': x1, 'len': torch.FloatTensor(question_len), 'labels': target_labels}
        return q_batch


    def train(self, training_data, full_question=False, create_runs=False, map_pattern=False) -> None:
        x_train, y_train, x_val, y_val, i_to_word, class_to_i, i_to_class = preprocess_dataset(training_data, full_question=full_question, create_runs=create_runs, map_pattern=map_pattern)
        self.class_to_i = class_to_i
        self.map_pattern = map_pattern
        self.use_wiki = use_wiki
        self.full_question = full_question

        self.i_to_class = i_to_class
        log = get(__name__, "dan.log")
        log.info('Batchifying data')
        i_to_word = ['<unk>', '<eos>'] + sorted(i_to_word)
        word_to_i = {x: i for i, x in enumerate(i_to_word)}
        self.word_to_i = word_to_i
        log.info('Vocab len: ' + str(len(self.word_to_i)))

        train_sampler = RandomSampler(list(zip(x_train, y_train)))
        dev_sampler = RandomSampler(list(zip(x_val, y_val)))
        dev_loader = DataLoader(list(zip(x_val, y_val)), batch_size=args.batch_size,
                                                   sampler=dev_sampler, num_workers=0,
                                                   collate_fn=self.batchify)
        train_loader = DataLoader(list(zip(x_train, y_train)), batch_size=args.batch_size,
                                           sampler=train_sampler, num_workers=0,
                                           collate_fn=self.batchify)

        self.model = DanModel(len(i_to_class), len(i_to_word))
        self.model = self.model.to(self.device)
        
        log.info(f'Loading GloVe')
        glove_word2idx, glove_vectors = load_glove("glove/glove.6B.300d.txt")
        for word, emb_index in word_to_i.items():
            if word.lower() in glove_word2idx:
                glove_index = glove_word2idx[word.lower()]
                glove_vec = torch.FloatTensor(glove_vectors[glove_index])
                glove_vec = glove_vec.cuda()
                self.model.text_embeddings.weight.data[emb_index, :].set_(glove_vec)


        log.info(f'Model:\n{self.model}')
        self.optimizer = Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, verbose=True, mode='max')


        temp_prefix = get_tmp_filename()
        self.model_file = f'{temp_prefix}.pt'

        print(f'Saving model to: {self.model_file}')
        log = get(__name__)
        manager = TrainingManager([
            BaseLogger(log_func=log.info), TerminateOnNaN(), EarlyStopping(monitor='test_acc', patience=10, verbose=1),
            MaxEpochStopping(100), ModelCheckpoint(create_save_model(self.model), self.model_file, monitor='test_acc')
        ])

        log.info('Starting training')

        epoch = 0
        while True:
            self.model.train()
            train_acc, train_loss, train_time = self.run_epoch(train_loader)

            self.model.eval()
            test_acc, test_loss, test_time = self.run_epoch(dev_loader, train=False)

            stop_training, reasons = manager.instruct(
                train_time, train_loss, train_acc,
                test_time, test_loss, test_acc
            )

            if stop_training:
                log.info(' '.join(reasons))
                break
            else:
                self.scheduler.step(test_acc)
            epoch += 1

    def run_epoch(self, data_loader, train=True):
        batch_accuracies = []
        batch_losses = []
        epoch_start = time.time()
        for idx, batch in tqdm(enumerate(data_loader)):
            x_batch = batch['text'].to(self.device)
            length_batch = batch['len'].to(self.device)
            y_batch = batch['labels'].to(self.device)
            if train:
                self.model.zero_grad()
            y_batch = y_batch.to(self.device)
            out = self.model(x_batch.to(self.device), length_batch.to(self.device))
            _, preds = torch.max(out, 1)
            accuracy = torch.mean(torch.eq(preds, y_batch).float()).data[0]
            batch_loss = self.criterion(out, y_batch)
            if train:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm(self.model.parameters(), .25)
                self.optimizer.step()
            batch_accuracies.append(accuracy)
            batch_losses.append(batch_loss.data[0])
        epoch_end = time.time()

        return np.mean(batch_accuracies), np.mean(batch_losses), epoch_end - epoch_start

    def guess(self, questions: List[str], max_n_guesses: Optional[int]) -> List[List[Tuple[str, float]]]:
        y_data = np.zeros((len(questions)))
        x_data = [tokenize_question(q, self.map_pattern) for q in questions]

        batches = self.batchify(list(zip(x_data, y_data)))
        guesses = []
        
        x_batch = batches["text"]
        length_batch = batches["len"]
        self.model.eval()
        out = self.model(x_batch.to(self.device), length_batch.to(self.device))
        probs = F.softmax(out).data.cpu().numpy()
        preds = np.argsort(-probs, axis=1)
        n_examples = probs.shape[0]
        for i in range(n_examples):
            example_guesses = []
            for p in preds[i][:max_n_guesses]:
                example_guesses.append((self.i_to_class[p], probs[i][p]))
            guesses.append(example_guesses)
        return guesses

    @classmethod
    def targets(cls) -> List[str]:
        return ['dan.pt', 'dan.pkl']

    @classmethod
    def load(cls, directory: str):
        with open(os.path.join(directory, 'dan.pkl'), 'rb') as f:
            params = cloudpickle.load(f)

        params['use_wiki'] = False # added to avoid model confusion for dan_pattern_es

        print('Params: Use Wiki: ' + str(params['use_wiki']) + ' Use Pattern: ' + str(params['map_pattern']) \
            + ' Wiki_links: ' + str(params['wiki_links']) + ' ES highlights: ' + str(params['use_es_highlight']))
        guesser = DanGuesser()
        guesser.class_to_i = params['class_to_i']
        guesser.i_to_class = params['i_to_class']
        guesser.word_to_i = params['word_to_i']
        guesser.device = params['device']
        guesser.map_pattern = params['map_pattern']
        guesser.wiki_links = params['wiki_links']
        guesser.use_wiki = params['use_wiki']
        guesser.use_es_highlight = params['use_es_highlight']
        guesser.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        guesser.model = DanModel(len(guesser.i_to_class), len(guesser.word_to_i))
        guesser.name = directory.split("/")[-1]
        guesser.model.load_state_dict(torch.load(
            os.path.join(directory, 'dan.pt'), map_location=lambda storage, loc: storage
        ).state_dict())
        guesser.model.eval()
        guesser.model = guesser.model.to(guesser.device)
        return guesser

    def save(self, directory: str) -> None:
        os.mkdir(directory)
        shutil.copyfile(self.model_file, os.path.join(directory, 'dan.pt'))
        with open(os.path.join(directory, 'dan.pkl'), 'wb') as f:
            cloudpickle.dump({
                'class_to_i': self.class_to_i,
                'i_to_class': self.i_to_class,
                'word_to_i': self.word_to_i,
                'use_wiki' : self.use_wiki,
                'device' : self.device,
                'map_pattern' : self.map_pattern,
                'wiki_links' : self.wiki_links,
                'use_es_highlight' : self.use_es_highlight
            }, f)

def create_app(path_dir=MODEL_PATH, enable_batch=True, predict=False):
    dan_guesser = DanGuesser.load(path_dir)
    app = Flask(__name__)

    @app.route('/api/1.0/quizbowl/act', methods=['POST'])
    def act():
        question = request.json['text']
        if predict:
            guess, buzz = guess_and_predict_buzz(dan_guesser, question)
        else:
            guess, buzz = guess_and_buzz(dan_guesser, question)
        return jsonify({'guess': guess, 'buzz': True if buzz else False})

    @app.route('/api/1.0/quizbowl/status', methods=['GET'])
    def status():
        return jsonify({
            'batch': enable_batch,
            'batch_size': 200,
            'ready': True
        })

    @app.route('/api/1.0/quizbowl/batch_act', methods=['POST'])
    def batch_act():
        questions = [q['text'] for q in request.json['questions']]
        if predict:
            return jsonify([
            {'guess': guess, 'buzz': True if buzz else False}
            for guess, buzz in batch_guess_and_predict_buzz(dan_guesser, questions)
            ])
        else:
            return jsonify([
            {'guess': guess, 'buzz': True if buzz else False}
            for guess, buzz in batch_guess_and_buzz(dan_guesser, questions)
            ])

    return app


@click.group()
def cli():
    pass


@cli.command()
@click.option('--host', default='0.0.0.0')
@click.option('--port', default=4861)
@click.option('--predict', default=True)
@click.option('--disable-batch', default=False, is_flag=True)
def web(host, port, disable_batch, predict):
    """
    Start web server wrapping tfidf model
    """
    app = create_app(enable_batch=not disable_batch, predict=predict)
    app.run(host=host, port=port, debug=False)


@cli.command()
@click.option('--use_wiki',is_flag=True, default=False)
@click.option('--full_question',is_flag=True, default=False)
@click.option('--create_runs',is_flag=True, default=False)
@click.option('--map_pattern',is_flag=True, default=False)
@click.option('--n_wiki_sentences', default=10)
@click.option('--category', default=None)
@click.option('--path', default="./")
def train(use_wiki, n_wiki_sentences, full_question, create_runs, category, map_pattern, path):
    """
    Train the tfidf model, requires downloaded data and saves to models/
    """
    dataset = QuizBowlDataset(guesser_train=True)
    training_data = dataset.training_data()

    if use_wiki and n_wiki_sentences > 0:
        print("Using wiki dataset with n_wiki_sentences: " + str(n_wiki_sentences))
        wiki_dataset = WikipediaDataset(set(training_data[1]), n_wiki_sentences)
        wiki_training_data = wiki_dataset.training_data()
        training_data[0].extend(wiki_training_data[0])
        training_data[1].extend(wiki_training_data[1])
   
    dan_guesser = DanGuesser()
    dan_guesser.train(training_data, full_question, create_runs)
    dan_guesser.save(path)


@cli.command()
@click.option('--model-path', default=MODEL_PATH)
def train_buzzer(model_path):
    """
    Train the tfidf buzzer saves to ./
    """
    dan_guesser = DanGuesser.load(model_path)
    dan_buzzer = Buzzer(dan_guesser)
    dan_buzzer.train()
    dan_buzzer.save(model_path)


@cli.command()
@click.option('--local-qanta-prefix', default='data/')
def download(local_qanta_prefix):
    """
    Run once to download qanta data to data/. Runs inside the docker container, but results save to host machine
    """
    util.download(local_qanta_prefix)


if __name__ == '__main__':
    cli()
