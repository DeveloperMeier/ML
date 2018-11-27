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

class BaseML:
	def __init__(self, seed, dataUrl, dataNames):
		self.seed = seed
		self.url = dataUrl
		self.names = dataNames
		# Load dataset
		self.data = pandas.read_csv(self.url, names=self.names)
		return self
    
	def encode_feature(self, labels, column):
		data = self.data
		for l in labels:
			data[l] = map(lambda x: int(x == l), data[column])
		del data[column]
		self.data = data
	
	
	def split_validation(self, x_start, x_end, y_end, validation_size = 0.20):
		array = self.data.values
		X = array[:,x_start:x_end]
		Y = array[:,y_end]
		return model_selection.train_test_split(X, Y, test_size=validation_size, random_state=self.seed)

	# Split-out validation dataset

	# Spot Check Algorithms
	def make_models(self, models = []):
		models.append(('LR(LogisticRegression):', LogisticRegression(solver="liblinear", multi_class="ovr")))
		models.append(('LDA(LinearDiscriminationAnalysis):', LinearDiscriminantAnalysis()))
		models.append(('KNN(KNeighborsClassifier):', KNeighborsClassifier()))
		models.append(('CART(DecisionTreeClassifier):', DecisionTreeClassifier()))
		models.append(('NB(GaussianNB):', GaussianNB()))
		models.append(('SVM(SVC):', SVC(gamma="auto")))
		return models


	def training_loop(self, X_train, Y_train, scoring, models=[], n_splits = 10):
		self.results = []
		self.names = []
		for name, model in models:
			kfold = model_selection.KFold(n_splits=n_splits, random_state=self.seed)
			cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
			self.results.append(cv_results)
			self.names.append(name)
			msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
			print(msg)
			return [self.results, self.names]


	def compare_models(self):
		# Compare Algorithms
		fig = plt.figure()
		fig.suptitle('Algorithm Comparison')
		ax = fig.add_subplot(111)
		plt.boxplot(self.results)
		ax.set_xticklabels(self.names)
		plt.show()

	def predict(self, X_train, Y_train, X_validation, Y_validation):
		# Make predictions on validation dataset
		knn = KNeighborsClassifier()
		knn.fit(X_train, Y_train)
		predictions = knn.predict(X_validation)
		print(accuracy_score(Y_validation, predictions))
		print(confusion_matrix(Y_validation, predictions))
		print(classification_report(Y_validation, predictions))

	def show(self, layout = (2,2)):
		print(self.data.shape)
		self.data.plot(kind='box', subplots=True, layout=layout, sharex=False, sharey=False)
		plt.show()
		self.data.hist()
		plt.show()
		scatter_matrix(self.data)
		plt.show()