# Hello, Machine Learning

This repo contains short machine learning projects using Scikit-learn's library.

## Logistic Regression

### Getting Started
I used Python 3.7.2, but this program should be backwards compatible to 3.4.

To run, simply install the requirements and run 'Log_Regression.py'.
```
pip3 install requirements.txt
python Log_Regression.py
```

'Log_Regression.py' does not take any arguments from the command line or require user input during execution.

### Input
I used the well-loved [20 Newsgroups](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html) dataset for this project. Each text sample in the dataset has a corresponding target category, such as 'soc.religion.christian' or 'rec.sport.baseball'.

### Output
Currently, I just output the accuracy score to the command line and produce a graphical confusion matrix. The confusion matrix shows the proportion of text samples from each category that were correctly and incorrectly classified by the Logistic Regression model.

### Resources
I didn't follow one particular tutorial; I picked through a few of them:
* From Scikit-learn's official documentation:
	* [sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
	* [Working With Text Data](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
	* [The 20 newsgroups text dataset](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html)
* [Classifying text with bag-of-words: a tutorial](http://fastml.com/classifying-text-with-bag-of-words-a-tutorial/) (Fast ML), which in turn references:
	* [Bag of Words Meets Bags of Popcorn](https://www.kaggle.com/c/word2vec-nlp-tutorial#part-1-for-beginners-bag-of-words) (Kaggle)
* [Multi-Class Text Classification with Scikit-Learn](https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f) (Towards Data Science)