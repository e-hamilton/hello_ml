import sys
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from Utilities.Spinner import *

MAX_CATEGORIES = 8

class Newsgroups:
	""" The Newsgroups class imports the 20 Newsgroups dataset from
	Scikit-learn and loads the text data into TF-IDF vectors.
	"""

	def __init__(self, categories):
		"""Constructor requires a categories array of 1-8 newsgroups. May 
		include any of the following:
		'alt.atheism',
		'comp.graphics',
		'comp.os.ms-windows.misc',
		'comp.sys.ibm.pc.hardware',
		'comp.sys.mac.hardware',
		'comp.windows.x',
		'misc.forsale',
		'rec.autos',
		'rec.motorcycles',
		'rec.sport.baseball',
		'rec.sport.hockey',
		'sci.crypt',
		'sci.electronics',
		'sci.med',
		'sci.space',
		'soc.religion.christian',
		'talk.politics.guns',
		'talk.politics.mideast',
		'talk.politics.misc',
		'talk.religion.misc'
		"""
		if len(categories) > MAX_CATEGORIES:
			err = ('Please limit categories[] to ' + str(MAX_CATEGORIES) 
					+ ' items or fewer.')
			raise ValueError(err)
		elif type(categories) is not list or len(categories) is 0:
			err = 'Must provide categories array.'
			raise ValueError(err)
		self.categories = categories
		self.train = None
		self.test = None
		self.target_names = None
		self.vocabulary = None


	def _fetchData(self):
		"""Downloads training and test datasets from remote host if not already 
		saved locally.

		For more info/optional args:
		https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html
		"""

		# Fetch training data (x,y)
		self.train = fetch_20newsgroups(subset='train',
										categories=self.categories,
										shuffle=True)
		# Fetch test data (x,y)
		self.test = fetch_20newsgroups(subset='test', 
									   categories=self.categories,
									   shuffle=True)
		# Each y-value (target) is an index in this array.
		self.target_names = list(self.train.target_names)


	def _formatData(self):
		"""Newsgroups.train.data and Newsgroups.test.data consist of text 
		documents which must be converted to numerical data. For the 20 
		Newsgroups dataset, Scikit recommends using the TF-IDF vectorizer to 
		turn the corpus into a matrix of TF-IDF features. Currently setting 
		n-gram range to (1,2), as any more requires too much memory.
		"""
		vectorizer = TfidfVectorizer(ngram_range=(1,2))
		self.train.x = vectorizer.fit_transform(self.train.data)
		self.train.y = self.train.target
		self.train.feature_count = self.train.x.shape[1]
		self.test.x = vectorizer.transform(self.test.data)
		self.test.y = self.test.target
		self.test.feature_count = self.test.x.shape[1]
		self.vocabulary = vectorizer.vocabulary_


	def load(self):
		"""Loads newsgroups specified in class constructor 
		(Newsgroups.categories) and vectorizes Newsgroups.train.x and 
		Newsgroups.test.x.

		Accessible after load() finished execution:
		* Newsgroups.target_names = list of strings; category names associated 
			with each target value
		* Newsgroups.train.x = vectorized training data
		* Newsgroups.train.y = corresponding targets for each document in 
			Newsgroups.train.x (indices in Newsgroups.target_names)
		* Newsgroups.test.x = vectorized test data
		* Newsgroups.test.y = corresponding targets for each document in 
			Newsgroups.test.x (indices in Newsgroups.target_names)
		* Newsgroups.train.feature_count = number of unique features (unigrams 
			and bigrams) in Newsgroups.train.x
		* Newsgroups.test.feature_count = number of unique features (unigrams 
			and bigrams) in Newsgroups.test.x
		* Newsgroups.vocabulary = dict; vocabulary created from TfidfVectorizer
		"""
		spinnerTask(self._fetchData, message="Loading data...")
		spinnerTask(self._formatData, message="Formatting data...")