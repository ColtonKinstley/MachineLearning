# Plot the decision boundry for linear least squares on some normally
# distributed 2-d catigorical data.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from sklearn import linear_model

# Create two Gaussian distributions in two dimensions centered at (1,1) and (5,5) then
# draw a few samples, plot them in orange and blue, classify the space with least
# squares regression. Then plot the least squares decision boundary in purple.


class Least_Squares_Binary_Classifier:
    def __init__(self):
        self.BLUE_MEAN = (3, 3)
        self.ORANGE_MEAN = (6, 6)
        self.BLUE_TRAIN_SIZE = 15
        self.BLUE_TEST_SIZE = 15
        self.ORANGE_TRAIN_SIZE = 15
        self.ORANGE_TEST_SIZE = 15

        self.train_new_model()

    def train_new_model(self):
        self.train_blue = np.array([
            list(x) for x in zip(
                np.random.normal(
                    loc=self.BLUE_MEAN[0], size=self.BLUE_TRAIN_SIZE),
                np.random.normal(
                    loc=self.BLUE_MEAN[1], size=self.BLUE_TRAIN_SIZE),
            )
        ])
        self.train_orange = np.array([
            list(x) for x in zip(
                np.random.normal(
                    loc=self.ORANGE_MEAN[0], size=self.ORANGE_TRAIN_SIZE),
                np.random.normal(
                    loc=self.ORANGE_MEAN[1], size=self.ORANGE_TRAIN_SIZE),
            )
        ])
        self.test_blue = np.array([
            list(x) for x in zip(
                np.random.normal(
                    loc=self.BLUE_MEAN[0], size=self.BLUE_TEST_SIZE),
                np.random.normal(
                    loc=self.BLUE_MEAN[1], size=self.BLUE_TEST_SIZE),
            )
        ])
        self.test_orange = np.array([
            list(x) for x in zip(
                np.random.normal(
                    loc=self.ORANGE_MEAN[0], size=self.ORANGE_TEST_SIZE),
                np.random.normal(
                    loc=self.ORANGE_MEAN[1], size=self.ORANGE_TEST_SIZE),
            )
        ])

        self.model = linear_model.LinearRegression()
        self.model.fit(
            np.concatenate((self.train_blue, self.train_orange)),
            np.concatenate((np.zeros(self.BLUE_TRAIN_SIZE),
                            np.ones(self.ORANGE_TRAIN_SIZE))),
        )

        # Determine the x1 and x2 intercepts in order to plot the decision
        # boundary which is 1/2 = Ax+b.
        self.x1 = (.5 - self.model.intercept_) / self.model.coef_[1]
        self.x2 = (.5 - self.model.intercept_) / self.model.coef_[0]

    def plot_results(self):
        fig1 = plt.figure()
        plt.scatter(
            x=self.train_blue[:, 0], y=self.train_blue[:, 1], color="blue")
        plt.scatter(
            x=self.train_orange[:, 0],
            y=self.train_orange[:, 1],
            color="orange")
        plt.scatter(
            x=self.test_blue[:, 0],
            y=self.test_blue[:, 1],
            color="blue",
            marker="+")
        plt.scatter(
            x=self.test_orange[:, 0],
            y=self.test_orange[:, 1],
            color="orange",
            marker="+",
        )
        plt.plot([self.x1, 0], [0, self.x2],
                 color="purple")  # plot intercepts with line connecting
        fig1.show()

    def get_errors(self):
        self.errors = []
        for index, prediction in enumerate(
                self.model.predict(
                    np.concatenate((self.train_blue, self.test_blue)))):
            if prediction > .5:
                self.errors.append(
                    np.concatenate((self.train_blue, self.test_blue))[index])

        for index, prediction in enumerate(
                self.model.predict(
                    np.concatenate((self.train_orange, self.test_orange)))):
            if prediction < .5:
                self.errors.append(
                    np.concatenate((self.train_orange,
                                    self.test_orange))[index])
        return self.errors
