from models.abalone import AbaloneML

seed = 7
dataUrl = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/abalone.csv"
dataNames = ['sex', 'length', 'diameter', 'height', 'weight', 'shucked-weight', 'viscera-weight', 'shell-weight', 'rings']
scoring = 'accuracy'


ml = AbaloneML(seed, dataUrl, dataNames)

X_train, X_validation, Y_train, Y_validation = ml.split_validation(x_start = 1, x_end = 7, y_end = 7)

model=ml.make_models(models = [])

results, names = ml.training_loop(X_train = X_train, Y_train = Y_train, scoring = scoring, models = model, n_splits = 10)

print(ml.predict(X_train = X_train, Y_train = Y_train, X_validation = X_validation, Y_validation = Y_validation))
ml.show(layout = (3,4))
preview = ml.data