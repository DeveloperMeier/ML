# Load libraries
import pandas
import matplotlib.pyplot as plt

# Pandas imports
from pandas.plotting      		   import scatter_matrix

# SKLearn Imports
from sklearn              		   import model_selection
from sklearn.metrics      		   import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model 		   import LogisticRegression
from sklearn.tree         		   import DecisionTreeClassifier
from sklearn.neighbors    		   import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes 		   import GaussianNB
from sklearn.svm 				   import SVC

from base.basicML import BaseML


class AbaloneML(BaseML):
	def __init__(self, seed, dataUrl, dataNames):
		BaseML.__init__(self, seed = seed, dataUrl = dataUrl, dataNames = dataNames)

	# This is gonna need overridden to work with a oneHotEncoder
	# def training_loop(self, X_train, Y_train, scoring, models=[], n_splits = 10):
		# return CheezML.training_loop(self, X_train, Y_train, scoring, models=[], n_splits=10)
		# self.results = []
		# self.names = []
		# for name, model in models:
		# 	kfold = model_selection.KFold(n_splits=n_splits, random_state=self.seed)
		# 	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
		# 	self.results.append(cv_results)
		# 	self.names.append(name)
		# 	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
		# 	print(msg)
		# 	return [self.results, self.names]

	