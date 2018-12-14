from typing import List, Optional, Tuple, Dict
from collections import defaultdict
import pickle
import json
from os import path
import os
import click
from tqdm import tqdm
from elasticsearch_dsl import DocType, Text, Keyword, Search, Index
from elasticsearch_dsl.connections import connections
import elasticsearch
from flask import Flask, jsonify, request

from jinja2 import Environment, PackageLoader, FileSystemLoader
from nltk.tokenize import word_tokenize

from qanta.util import *
from qanta.dataset import QuizBowlDataset
from qanta.preprocess import WikipediaDataset
from collections import namedtuple
WikipediaPage = namedtuple('WikipediaPage', ['id', 'title', 'text', 'url'])

log = get(__name__)
ES_PARAMS = 'es_params.pickle'
connections.create_connection(hosts=['qb'])
BUZZ_NUM_GUESSES = 10
BUZZ_THRESHOLD = 0.2


def create_doctype(index_name, similarity):
    if similarity == 'default':
        wiki_content_field = Text()
        qb_content_field = Text()
    else:
        wiki_content_field = Text(similarity=similarity)
        qb_content_field = Text(similarity=similarity)

    class Answer(DocType):
        page = Text(fields={'raw': Keyword()})
        wiki_content = wiki_content_field
        qb_content = qb_content_field

        class Meta:
            index = index_name

    return Answer

class ElasticSearchIndex:
    def __init__(self, name='qb', similarity='default', bm25_b=None, bm25_k1=None):
        self.name = name
        self.ix = Index(self.name)
        self.answer_doc = create_doctype(self.name, similarity)
        if bm25_b is None:
            bm25_b = .75
        if bm25_k1 is None:
            bm25_k1 = 1.2
        self.bm25_b = bm25_b
        self.bm25_k1 = bm25_k1

    def delete(self):
        try:
            self.ix.delete()
        except elasticsearch.exceptions.NotFoundError:
            log.info('Could not delete non-existent index.')

    def exists(self):
        return self.ix.exists()

    def init(self):
        self.ix.create()
        self.ix.close()
        self.ix.put_settings(body={'similarity': {
            'qb_bm25': {'type': 'BM25', 'b': self.bm25_b, 'k1': self.bm25_k1}}
        })
        self.ix.open()
        self.answer_doc.init(index=self.name)

    def build_large_docs(self, documents: Dict[str, str], use_wiki=True, use_qb=True, rebuild_index=False):
        if rebuild_index and bool(int(os.getenv('QB_REBUILD_INDEX', 0))):
            log.info(f'Deleting index: {self.name}')
            self.delete()

        if self.exists():
            log.info(f'Index {self.name} exists')
        else:
            log.info(f'Index {self.name} does not exist')
            self.init()
            with open("data/wiki_lookup.json", 'rb') as f:
                raw_lookup: Dict[str, Dict] = json.load(f)
                wiki_lookup: Dict[str, WikipediaPage] = {
                title: WikipediaPage(page['id'], page['title'], page['text'], page['url'])
                for title, page in raw_lookup.items()
            }
            log.info('Indexing questions and corresponding wikipedia pages as large docs...')
            for page in tqdm(documents):
                if use_wiki and page in wiki_lookup:
                    wiki_content = wiki_lookup[page].text
                else:
                    wiki_content = ''

                if use_qb:
                    qb_content = documents[page]
                else:
                    qb_content = ''

                answer = self.answer_doc(
                    page=page,
                    wiki_content=wiki_content, qb_content=qb_content
                )
                answer.save(index=self.name)

    def build_many_docs(self, pages, documents, use_wiki=True, use_qb=True, rebuild_index=False):
        if rebuild_index and bool(int(os.getenv('QB_REBUILD_INDEX', 0))):
            log.info(f'Deleting index: {self.name}')
            self.delete()

        if self.exists():
            log.info(f'Index {self.name} exists')
        else:
            log.info(f'Index {self.name} does not exist')
            self.init()
            log.info('Indexing questions and corresponding pages as many docs...')
            if use_qb:
                log.info('Indexing questions...')
                for page, doc in tqdm(documents):
                    self.answer_doc(page=page, qb_content=doc).save()

            if use_wiki:
                log.info('Indexing wikipedia...')
                wiki_lookup = WikipediaDataset()
                for page in tqdm(pages):
                    if page in wiki_lookup:
                        content = word_tokenize(wiki_lookup[page].text)
                        for i in range(0, len(content), 200):
                            chunked_content = content[i:i + 200]
                            if len(chunked_content) > 0:
                                self.answer_doc(page=page, wiki_content=' '.join(chunked_content)).save()

    def search(self, text: str, max_n_guesses: int,
               normalize_score_by_length=False,
               wiki_boost=1, qb_boost=1):
        if not self.exists():
            raise ValueError('The index does not exist, you must create it before searching')

        if wiki_boost != 1:
            wiki_field = 'wiki_content^{}'.format(wiki_boost)
        else:
            wiki_field = 'wiki_content'

        if qb_boost != 1:
            qb_field = 'qb_content^{}'.format(qb_boost)
        else:
            qb_field = 'qb_content'

        s = Search(index=self.name)[0:max_n_guesses].query(
            'multi_match', query=text, fields=[wiki_field, qb_field]
        )
        results = s.execute()
        guess_set = set()
        guesses = []
        if normalize_score_by_length:
            query_length = len(text.split())
        else:
            query_length = 1

        for r in results:
            if r.page in guess_set:
                continue
            else:
                guesses.append((r.page, r.meta.score / query_length))
        return guesses

class ElasticSearchGuesser():
    def __init__(self):
        super().__init__()
        self.n_cores = 4
        self.use_wiki = True
        self.use_qb = True
        self.many_docs = False
        self.normalize_score_by_length = True
        self.qb_boost = 1
        self.wiki_boost = 1
        self.similarity_name = "BM25"
        if self.similarity_name == 'BM25':
            self.similarity_k1 = 2.0
            self.similarity_b = 0.75
        else:
            self.similarity_k1 = None
            self.similarity_b = None
        self.index = ElasticSearchIndex(
            name='qb', similarity=self.similarity_name,
            bm25_b=self.similarity_b, bm25_k1=self.similarity_k1
        )

    def guess_and_buzz(self, question_text) -> Tuple[str, bool]:
        guesses = self.guess([question_text], BUZZ_NUM_GUESSES)[0]
        scores = [guess[1] for guess in guesses]
        buzz = scores[0] / sum(scores) >= BUZZ_THRESHOLD
        return guesses[0][0], buzz


    def batch_guess_and_buzz(self, questions) -> List[Tuple[str, bool]]:
        question_guesses = [self.guess_and_buzz([questions[i]]) for i in range(len(questions))]
        outputs = []
        for guesses in question_guesses:
            scores = [guess[1] for guess in guesses]
            buzz = scores[0] / sum(scores) >= BUZZ_THRESHOLD
            outputs.append((guesses[0][0], buzz))
        # print(outputs)
        return outputs


    def train(self, training_data):
        if self.many_docs:
            pages = set(training_data[1])
            documents = []
            for sentences, page in zip(training_data[0], training_data[1]):
                paragraph = ' '.join(sentences)
                documents.append((page, paragraph))
            self.index.build_many_docs(
                pages, documents,
                use_qb=self.use_qb, use_wiki=self.use_wiki, rebuild_index=True
            )
        else:
            documents = {}
            for sentences, page in zip(training_data[0], training_data[1]):
                paragraph = ' '.join(sentences)
                if page in documents:
                    documents[page] += ' ' + paragraph
                else:
                    documents[page] = paragraph

            self.index.build_large_docs(
                documents,
                use_qb=self.use_qb,
                use_wiki=self.use_wiki,
                rebuild_index=True
            )

    def guess(self, questions: List[str], max_n_guesses: Optional[int]):
        def es_search(query):
            return self.index.search(
                query, max_n_guesses,
                normalize_score_by_length=self.normalize_score_by_length,
                wiki_boost=self.wiki_boost, qb_boost=self.qb_boost
            )

        if len(questions) > 1:
            sc = create_spark_context(configs=[('spark.executor.cores', self.n_cores), ('spark.executor.memory', '20g')])
            return sc.parallelize(questions, 16 * self.n_cores).map(es_search).collect()
        elif len(questions) == 1:
            return [es_search(questions[0])]
        else:
            return []

    @classmethod
    def targets(cls):
        return [ES_PARAMS]

    @classmethod
    def load(cls):
        with open(ES_PARAMS, 'rb') as f:
            params = pickle.load(f)
        guesser = ElasticSearchGuesser()
        guesser.n_cores = params['n_cores']
        guesser.use_wiki = params['use_wiki']
        guesser.use_qb = params['use_qb']
        guesser.many_docs = params['many_docs']
        guesser.normalize_score_by_length = params['normalize_score_by_length']
        guesser.qb_boost = params['qb_boost']
        guesser.wiki_boost = params['wiki_boost']
        guesser.similarity_name = params['similarity_name']
        guesser.similarity_b = params['similarity_b']
        guesser.similarity_k1 = params['similarity_k1']

        return guesser

    def save(self):
        with open(ES_PARAMS, 'wb') as f:
            pickle.dump({
                'n_cores': self.n_cores,
                'use_wiki': self.use_wiki,
                'use_qb': self.use_qb,
                'many_docs': self.many_docs,
                'normalize_score_by_length': self.normalize_score_by_length,
                'qb_boost': self.qb_boost,
                'wiki_boost': self.wiki_boost,
                'similarity_name': self.similarity_name,
                'similarity_k1': self.similarity_k1,
                'similarity_b': self.similarity_b
            }, f)

    def web_api(self, host='0.0.0.0', port=4861, debug=True, enable_batch=False):

        app = Flask(__name__)

        @app.route('/api/1.0/quizbowl/act', methods=['POST'])
        def act():
            question = request.json['text']
            guess, buzz = self.guess_and_buzz(question)
            return jsonify({'guess': guess, 'buzz': True if buzz else False})

        @app.route('/api/1.0/quizbowl/status', methods=['GET'])
        def status():
            print("Enable: " + str(enable_batch))
            return jsonify({
                'batch': enable_batch,
                'batch_size': 200,
                'ready': True
            })

        @app.route('/api/1.0/quizbowl/batch_act', methods=['POST'])
        def batch_act():
            questions = [q['text'] for q in request.json['questions']]
            return jsonify([
                {'guess': guess, 'buzz': True if buzz else False}
                for guess, buzz in self.batch_guess_and_buzz(questions)
            ])

        @app.route('/api/1.0/quizbowl/get_highlights', methods=['POST'])
        def get_highlights():
            wiki_field = 'wiki_content'
            qb_field = 'qb_content'
            text = request.json['text']
            s = Search(index='qb')[0:10].query(
                'multi_match', query=text, fields=[wiki_field, qb_field])
            s = s.highlight(wiki_field).highlight(qb_field)
            results = list(s.execute())

            if len(results) == 0:
                highlights = {'wiki': [''],
                              'qb': [''],
                              'guess': ''}
            else:
                guess = results[0] # take the best answer
                _highlights = guess.meta.highlight 
                try:
                    wiki_content = list(_highlights.wiki_content)
                except AttributeError:
                    wiki_content = ['']

                try:
                    qb_content = list(_highlights.qb_content)
                except AttributeError:
                    qb_content = ['']

                highlights = {'wiki': wiki_content,
                              'qb': qb_content,
                              'guess': guess.page}
            return jsonify(highlights)

        app.run(host=host, port=port, debug=debug, use_reloader=False)


@click.group()
def cli():
    pass


@cli.command()
@click.option('--host', default='0.0.0.0')
@click.option('--port', default=4861)
@click.option('--disable-batch', default=False, is_flag=True)
def web(host, port, disable_batch):
    """
    Start web server wrapping tfidf model
    """
    elastic_guesser = ElasticSearchGuesser()
    elastic_guesser.web_api(host, port, enable_batch=False)

@cli.command()
@click.option('--use_wiki',is_flag=True, default=False)
@click.option('--n_wiki_sentences', default=10)
def train(use_wiki, n_wiki_sentences):
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
   
    elastic_guesser = ElasticSearchGuesser()
    elastic_guesser.train(training_data)
    elastic_guesser.web_api()

@cli.command()
@click.option('--local-qanta-prefix', default='data/')
def download(local_qanta_prefix):
    """
    Run once to download qanta data to data/. Runs inside the docker container, but results save to host machine
    """
    util.download(local_qanta_prefix)


if __name__ == '__main__':
    cli()
