from tqdm import tqdm
import pandas as pd
from typing import List, Optional, Tuple
from collections import defaultdict
import pickle
import json
from os import path
import xgboost
import numpy as np
from qanta import util
from qanta.dataset import QuizBowlDataset

from multiprocessing import Pool
from functools import partial

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def generate_guesses(model, max_guesses=10, char_skip=50,full_question=False, first_sentence=False) -> pd.DataFrame:
		"""
		Generates guesses for this guesser for all questions in specified folds and returns it as a
		DataFrame

		WARNING: this method assumes that the guesser has been loaded with load or trained with
		train. Unexpected behavior may occur if that is not the case.
		:param max_n_guesses: generate at most this many guesses per question, sentence, and token
		:param folds: which folds to generate guesses for
		:param char_skip: generate guesses every 10 characters
		:return: dataframe of guesses
		"""
		if full_question and first_sentence:
			raise ValueError('Invalid option combination')

		dataset = QuizBowlDataset(guesser_train=False,buzzer_train=True)
		questions_by_fold = dataset.questions_by_fold()
		
		folds = ['buzztrain']
		
		q_folds = []
		q_qnums = []
		q_char_indices = []
		question_texts = []
		q_score_and_guess = []
		q_answers = []
	
		for fold in folds:
			questions = questions_by_fold[fold]
			print("# of questions: " + str(len(questions)))
			for q in tqdm(questions):
				if full_question:
					q_folds.append(fold)
					question_texts.append(q.text)
					q_qnums.append(q.qanta_id)
					q_char_indices.append(len(q.text))
					q_answers.append(q.page)
				elif first_sentence:
					q_folds.append(fold)
					question_texts.append(q.first_sentence)
					q_qnums.append(q.qanta_id)
					q_char_indices.append(q.tokenizations[0][1])
					q_answers.append(q.page)
				else:
					curr_ques = []
					for text_run, char_ix in zip(*(q.runs(char_skip))):
						q_folds.append(fold)
						question_texts.append(text_run)
						curr_ques.append(text_run)
						q_qnums.append(q.qanta_id)
						q_answers.append(q.page)
						q_char_indices.append(char_ix)
					guesses = model.guess(curr_ques, max_guesses)
					for guess in guesses:
						q_score_and_guess.append(guess)

		return pd.DataFrame({
			'qanta_id': q_qnums,
			'char_index': q_char_indices,
			'fold': q_folds,
			'score_and_guess': q_score_and_guess,
			'answers' : q_answers
		})


class Buzzer():
	"""Buzzer"""
	def __init__(self, model, max_n_guesses=10):
		self.model = model
		questions = QuizBowlDataset(buzzer_train=True).questions_by_fold()
		self.questions = {q.qanta_id: q for q in questions['buzztrain']}
		self.max_n_guesses = max_n_guesses
		self.df = generate_guesses(self.model, self.max_n_guesses)

		# self.df.to_pickle("df.pkl")

		print("# of Train: " + str(len(self.df)))
		self.create_features()

	def create_features(self):
		df_groups = self.df.groupby('qanta_id')

		xs = []
		ys = []
		for qid, q_rows in tqdm(df_groups):
			qid, vectors, char_indices, labels = self.process_question(qid, q_rows)
			for i in range(len(labels)):
				xs.append(np.append(vectors[i][:10], char_indices[i]))
				ys.append(labels[i])
				
		self.xs = np.asarray(xs)
		self.ys = np.asarray(ys)

	def process_question(self, qid, q_rows):
		'''multiprocessing worker that converts the guesser output of a single
			question into format used by the buzzer
		'''
		qid = q_rows.qanta_id.tolist()[0]
		answer = self.questions[qid].page
		q_rows = q_rows.groupby('char_index')
		char_indices = sorted(q_rows.groups.keys())
		guesses_sequence = []
		labels = []
		for idx in char_indices:
			p = q_rows.get_group(idx)
			guesses_sequence.append(list(p.score_and_guess))
			labels.append(int(p.score_and_guess.tolist()[0][0][0] == answer))
		vectors = self.vector_converter(guesses_sequence)
		return qid, vectors, char_indices, labels


	def vector_converter(self, guesses_sequence):
		'''vector converter / feature extractor with only prob

		Args:
			guesses_sequence: a sequence (length of question) of list of guesses
				(n_guesses), each entry is (guess, prob)
		Returns:
			a sequence of vectors
		'''
		length = len(guesses_sequence)
		prev_prob_vec = [0. for _ in range(self.max_n_guesses)]
		prev_dict = dict()

		vecs = []
		for i in range(length):
			prob_vec = []
			prob_diff_vec = []
			isnew_vec = []
			guesses = guesses_sequence[i][0]
			for guess, prob in guesses:
				prob_vec.append(prob)
				if i > 0 and guess in prev_dict:
					prev_prob = prev_dict[guess]
					prob_diff_vec.append(prob - prev_prob)
					isnew_vec.append(0)
				else:
					prob_diff_vec.append(prob)
					isnew_vec.append(1)
			if len(guesses) < self.max_n_guesses:
				for k in range(max(self.max_n_guesses - len(guesses), 0)):
					prob_vec.append(0)
					prob_diff_vec.append(0)
					isnew_vec.append(0)
			features = prob_vec[:10] \
				+ isnew_vec[:10] \
				+ prob_diff_vec[:10] 
			vecs.append(np.array(features, dtype=np.float32))
			prev_prob_vec = prob_vec
			prev_dict = {g: p for g, p in guesses}
		return vecs

	def train(self):

		seed = 7
		test_size = 0.33
		X_train, X_test, y_train, y_test = train_test_split(self.xs, self.ys, test_size=test_size, random_state=seed)

		# fit model no training data
		self.buzzer_model = XGBClassifier()
		self.buzzer_model.fit(X_train, y_train)

		# make predictions for test data
		y_pred = self.buzzer_model.predict(X_test)

		# evaluate predictions
		accuracy = accuracy_score(y_test, y_pred)
		print("Accuracy: %.2f%%" % (accuracy * 100.0))

	def save(self,save_path):
		pickle.dump(self.buzzer_model, open(save_path + "buzzer_"+ self.model.name + ".pkl", "wb"))