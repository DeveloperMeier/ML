# Load libraries
import pandas
import matplotlib.pyplot as plt
import numpy as np

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

from base.basic_ml import BaseML


class AbaloneML(BaseML):
	def __init__(self, seed, dataUrl, dataNames):
		return BaseML.__init__(self, seed = seed, dataUrl = dataUrl, dataNames = dataNames).encode_feature('MFI', 'sex')
       
        
	def split_validation(self, x_start, x_end, y_end, validation_size = 0.20):                
		X = self.data.values
		Y = self.data.rings.values
		del self.data['rings']
		return model_selection.train_test_split(X, Y, test_size=validation_size, random_state=self.seed)

	