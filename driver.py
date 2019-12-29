from neuralnet.neuralnet import NeuralNet
from sklearn import model_selection
import pandas as pd
import time


if __name__ == "__main__":
    # DataSet Downloaded from UCI Machine Learning dataset repository
    # Download dataset from the link https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/


    #parameters for tuning, change these parameters for different outputs
    learning_rate = 0.5
    max_iterations = 10000

    df = pd.read_csv("data/breast-cancer-2.csv", header=None)
    trainDF, testDF = model_selection.train_test_split(df, test_size=0.20)

    #To run code with the dataset provided in the assignment, uncomment next two lines and comment above two lines
    # trainDF = pd.read_csv("dataset/train.csv")
    # testDf = pd.read_csv("dataset/test.csv")

    neural_network = NeuralNet(trainDF)

    # Code runs on all three activation functions to easily compare the outputs.
    # The activation variable is configurable. To run the code with only one activation function,
    #comment the other two.

    start = time.time()
    neural_network.train(activation="sigmoid", learning_rate=learning_rate, max_iterations=max_iterations)
    testError = neural_network.predict(testDF, activation="sigmoid")
    elapsed = (time.time() - start)

    print("By Using learning rate as %s and max_iterations as %s and processing time as %s" % (learning_rate, max_iterations, elapsed))
    print("test error sum for sigmoid activation function ", testError)

    print("*****************************************************")

    start = time.time()
    neural_network.train(activation="tanh", learning_rate=learning_rate, max_iterations=max_iterations)
    testError = neural_network.predict(testDF, activation="tanh")
    elapsed = (time.time() - start)

    print("By Using learning rate as %s and max_iterations as %s and processing time as %s" % (learning_rate, max_iterations, elapsed))
    print("test error sum for tanh activation function ", testError)

    print("*****************************************************")

    start = time.time()
    neural_network.train(activation="relu", learning_rate=learning_rate, max_iterations=max_iterations)
    testError = neural_network.predict(testDF, activation="relu")
    elapsed = (time.time() - start)

    print("By Using learning rate as %s and max_iterations as %s and processing time as %s" % (learning_rate, max_iterations, elapsed))
    print("test error sum for relu activation function ", testError)