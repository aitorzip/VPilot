from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras.callbacks import EarlyStopping
from keras import optimizers
import random
import numpy
import nnmodel
import utils

# The model from ai-tor
def create_model(optimizer='Adam' , init='normal', lr=0.01, reg_lambda=0.0):
    model = nnmodel.getNNModel(reg_lambda=reg_lambda)
    # other parameters can be set by adding more if statements if needed
    methodToCall = getattr(optimizers, optimizer)
    model.compile(optimizer=methodToCall(lr=lr), loss="mse")
    #model.compile()
    return model

# Randomn search could also be implemented but let's test this first
def grid_search(x,y):
    seed = random.randint(0, 4294967295)
    numpy.random.seed(seed)
    # create model
    model = KerasRegressor(build_fn=create_model, verbose=0)
    stopping_callback = EarlyStopping(patience=5)
    fit_params = dict(callbacks=[stopping_callback], validation_split=0.4, batch_size=100, nb_epoch=10)
    # grid search epochs, batch size and optimizer
    # feel free to adjust this stuffs to test more than I have here
    optimizers = ['RMSprop', 'Adam']
    #init = ['glorot_uniform', 'normal', 'uniform', 'lecun_uniform', 'zero'] #not used for now
    lr = [0.01,0.05,0.1]
    reg_lambda = [0, 0.01, 0.1, 0.2, 0.5, 1]

    param_grid = dict(lr=lr, optimizer=optimizers, reg_lambda=reg_lambda)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, fit_params=fit_params)
    grid_result = grid.fit(x, y)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    # feel free to uncomment if you can afford to print all the results on your screen. Warning! It's gonna be pretty long
    #for params, mean_score, scores in grid_result.grid_scores_:
    #    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))


#test

data = utils.load_randomized_udacity_dataset("/media/aitor/Data/udacity/dataset-croped.bag")
grid_search(data[0], data[1])