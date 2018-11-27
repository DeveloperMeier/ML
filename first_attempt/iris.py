from base.basic_ml import BaseML

seed = 7
dataUrl = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
dataNames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
scoring = 'accuracy'

ml = BaseML(seed, dataUrl, dataNames)

X_train, X_validation, Y_train, Y_validation = ml.split_validation(x_start = 0, x_end = 4, y_end = 4)

model=ml.make_models(models = [])

results, names = ml.training_loop(X_train = X_train, Y_train = Y_train, scoring = scoring, models = model, n_splits = 10)

ml.predict(X_train = X_train, Y_train = Y_train, X_validation = X_validation, Y_validation = Y_validation)
ml.show(layout = (2,2))
