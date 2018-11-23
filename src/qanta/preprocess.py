import re
from typing import List
import string
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
import numpy
import random
import json
# from unidecode import unidecode
from typing import Set
import nltk
import re

ftp_patterns = {
    '\n',
    ', for 10 points,',
    ', for ten points,',
    '--for 10 points--',
    'for 10 points, ',
    'for 10 points--',
    'for ten points, ',
    'for 10 points ',
    'for ten points ',
    ', ftp,'
    'ftp,',
    'ftp'
}

patterns = ftp_patterns | set(string.punctuation)
regex_pattern = '|'.join([re.escape(p) for p in patterns])
regex_pattern += r'|\[.*?\]|\(.*?\)'

def extract_wiki_sentences(title, text, n_sentences, replace_title_mentions=''):
    """
    Extracts the first n_paragraphs from the text of a wikipedia page corresponding to the title.
    strip_title_mentions and replace_title_mentions control handling of references to the title in text.
    Oftentimes QA models learn *not* to answer entities mentioned in the question so this helps deal with this
    in the domain adaptation case.

    :param title: title of page
    :param text: text of page
    :param n_paragraphs: number of paragraphs to use
    :param replace_title_mentions: Replace mentions with the provided string token, by default removing them
    :return:
    """
    # Get simplest representation of title and text
    title = str(title).replace('_', ' ')
    text = str(text)

    # Split on non-alphanumeric
    title_words = re.split('[^a-zA-Z0-9]', title)
    title_word_pattern = '|'.join(re.escape(w.lower()) for w in title_words)

    # Breaking by newline yields paragraphs. Ignore the first since its always just the title
    paragraphs = [p for p in text.split('\n') if len(p) != 0][1:]
    sentences = []
    for p in paragraphs:
        formatted_text = re.sub(title_word_pattern, replace_title_mentions, p, flags=re.IGNORECASE)
        # Cleanup whitespace
        formatted_text = re.sub('\s+', ' ', formatted_text).strip()

        sentences.extend(nltk.sent_tokenize(formatted_text))

    return sentences[:n_sentences]

class WikipediaDataset():
    def __init__(self, answers: Set[str], n_sentences=5, replace_title_mentions=''):
        super().__init__()
        self.answers = answers
        self.n_sentences = n_sentences
        self.replace_title_mentions = replace_title_mentions

    def training_data(self):
        wiki_content = []
        wiki_answers = []
        wiki_lookup = None
        with open("data/wiki_lookup.json") as f:
            wiki_lookup = json.load(f)
        for ans in self.answers:
#             wiki_page = wikipedia.page( unidecode(ans).replace('_', ' '))
            if ans not in wiki_lookup:
                continue
            wiki_page = wiki_lookup[ans]
            if len(wiki_page["text"]) != 0:
                sentences = extract_wiki_sentences(
                    ans, wiki_page["text"], self.n_sentences,
                    replace_title_mentions=self.replace_title_mentions
                )
                for sent in sentences:
                    wiki_content.append([sent])
                    wiki_answers.append(ans)

        return wiki_content, wiki_answers, None

def clean_question(question: str):
    """
    Remove pronunciation guides and other formatting extras
    :param question:
    :return:
    """
    return re.sub(regex_pattern, '', question.strip().lower())


def tokenize_question(text: str) -> List[str]:
    return word_tokenize(clean_question(text))


def format_guess(guess):
    return guess.strip().lower().replace(' ', '_').replace(':', '').replace('|', '')

def preprocess_dataset(data, train_size=.9, test_size=.1,
                       vocab=None, class_to_i=None, i_to_class=None,
                       create_runs=False, full_question=False):
    """
    This function does primarily text preprocessing on the dataset. It will return x_train and x_test as a list of
    examples where each word is a tokenized word list (not padded). y_train and y_test is a list of indices coresponding
    to the class labels that are associated with i_to_class and class_to_i. vocab consists of any word which occurred
    in the training set.
    
    TODO: Implement an option for maximum vocab size which takes the most frequently occurring words only.
    
    :param data: 
    :param train_size: 
    :param vocab: 
    :param class_to_i: 
    :param i_to_class: 
    :param create_runs: 
    :param full_question: 
    :return:
    """
    if full_question and create_runs:
        raise ValueError('The options create_runs={} and full_question={} are not compatible'.format(
            create_runs, full_question))
    if train_size + test_size > 1:
        raise ValueError(
            f'Train + test must sum to 1 or less: train={train_size} test={test_size} sum={train_size + test_size}')

    classes = set(data[1])
    if class_to_i is None or i_to_class is None:
        class_to_i = {}
        i_to_class = []
        for i, ans_class in enumerate(classes):
            class_to_i[ans_class] = i
            i_to_class.append(ans_class)

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    if vocab is None:
        vocab = set()

    question_runs_with_answer = list(zip(data[0], data[1]))
    if train_size != 1:
        train, test = train_test_split(question_runs_with_answer, train_size=train_size, test_size=test_size)
    else:
        train = question_runs_with_answer
        test = []

    for q, ans in train:
        q_text = []
        for sentence in q:
            t_question = tokenize_question(sentence)
            if create_runs or full_question:
                q_text.extend(t_question)
            else:
                q_text = t_question
            if len(t_question) > 0:
                for w in t_question:
                    vocab.add(w)
                if create_runs:
                    x_train.append(list(q_text))
                elif not full_question:
                    x_train.append(q_text)

                if not full_question:
                    y_train.append(class_to_i[ans])
        if full_question:
            x_train.append(q_text)
            y_train.append(class_to_i[ans])

    for q, ans in test:
        q_text = []
        for sentence in q:
            t_question = tokenize_question(sentence)
            if len(t_question) > 0:
                if create_runs or full_question:
                    q_text.extend(t_question)
                    if not full_question:
                        x_test.append(list(q_text))
                else:
                    q_text = t_question
                    x_test.append(q_text)
                if not full_question:
                    y_test.append(class_to_i[ans])
        if full_question:
            x_test.append(q_text)
            y_test.append(class_to_i[ans])

    return (x_train, y_train,
            x_test, y_test,
            vocab, class_to_i, i_to_class)
