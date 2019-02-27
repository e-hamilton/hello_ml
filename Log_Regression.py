"""
This is a basic Logistic Regression program I wrote in an effort to dip my
toes into Scikit-learn's machine learning tools. It evaluates unigrams and
bigrams in several categories of the 20 Newsgroups dataset and produces a
confusion matrix.

This project doesn't follow a single tutorial; I looked at several and picked
elements from each. The resources I used are listed at the end of this file.

-Emily Hamilton, February 2019
"""

import sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from numpy import newaxis
import matplotlib.pyplot as plt
import seaborn as sns
from Utilities.Spinner import *
from Utilities.Newsgroups import *

def LR_run(solver, dataset):
	# Train Logistic Regression-- can test different solvers, but I believe
	# this classification problem warrants a 'multi_class=multinomial'
	# setting per:
	# https://scikit-learn.org/stable/modules/multiclass.html
	LR = LogisticRegression(solver=solver, multi_class='multinomial')
	LR.fit(dataset.train.x, dataset.train.y)
	predicted = LR.predict(dataset.test.x)
	return LR, predicted


if __name__ == "__main__":
	# Get training and testing data
	categories = ['alt.atheism', 'soc.religion.christian',
					'comp.graphics', 'sci.electronics',
					'rec.autos', 'rec.motorcycles', 
					'rec.sport.baseball', 'rec.sport.hockey']
	dataset = Newsgroups(categories)
	dataset.load()

	# Run Logistic Regression to get LR model and prediction
	LR, predicted = spinnerTask(LR_run, 'lbfgs', dataset, 
								message=('Modeling data (this may take a few ' +
								'minutes)...'))

	total_score = LR.score(dataset.test.x, dataset.test.y)
	print("\nTotal Score:", total_score)

	# Produce Confusion Matrix
	c_matrix = confusion_matrix(dataset.test.y, predicted)
	c_matrix_norm = c_matrix.astype('float') / c_matrix.sum(axis=1)[:, newaxis]
	fig, ax = plt.subplots(figsize=(10,10))
	palette = sns.cubehelix_palette(40, start=1, rot=-.75, dark=.2, light=.95)
	sns.heatmap(c_matrix_norm, annot=True, xticklabels=dataset.target_names,
				yticklabels=dataset.target_names, cmap=palette)
	plt.title('Logistic Regression Confusion Matrix')
	plt.ylabel('Actual Classification')
	plt.xlabel('Predicted Classification')
	plt.show()

"""
RESOURCES:
* https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
* https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
* https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html
* https://www.kaggle.com/c/word2vec-nlp-tutorial#part-1-for-beginners-bag-of-words
* http://fastml.com/classifying-text-with-bag-of-words-a-tutorial/
* https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f
"""