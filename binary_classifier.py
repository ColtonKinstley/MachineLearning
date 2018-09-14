"""
binary_classifier.py
====================
Plot the decision boundry for linear least squares on some normally distributed
2-d catigorical data. Specifically create two Gaussian distributions in two
dimensions centered at BLUE_MEAN and ORANGE_MEAN then draw a few samples, plot
them in orange and blue, classify the space with least squares regression. Then
plot the least squares decision boundary in purple.

"""
import matplotlib.pyplot as plt
import numpy as np

from sklearn import linear_model

# import pandas as pd
# import scipy
# import seaborn as sns


class Binary_Classifier:
    def __init__(self):
        self.BLUE_MEAN = (3, 3)
        self.ORANGE_MEAN = (5, 5)
        self.BLUE_TRAIN_SIZE = 15
        self.BLUE_SIZE = 30
        self.ORANGE_TRAIN_SIZE = 15
        self.ORANGE_SIZE = 30

        self.generate_new_sample()
        self.train_new_model()

    def generate_new_sample(self):
        """generate_new_sample"""
        self.blue_set = np.array([
            list(x) for x in zip(
                np.random.normal(loc=self.BLUE_MEAN[0], size=self.BLUE_SIZE),
                np.random.normal(loc=self.BLUE_MEAN[1], size=self.BLUE_SIZE),
            )
        ])

        self.orange_set = np.array([
            list(x) for x in zip(
                np.random.normal(
                    loc=self.ORANGE_MEAN[0], size=self.ORANGE_SIZE),
                np.random.normal(
                    loc=self.ORANGE_MEAN[1], size=self.ORANGE_SIZE),
            )
        ])

    def train_new_model(self):

        np.random.shuffle(self.blue_set)  # put the points in a random order

        # take the first part of the shuffled list for traning and last part for test
        self.blue_training_set = self.blue_set[0:self.BLUE_TRAIN_SIZE]
        self.blue_test_set = self.blue_set[self.BLUE_TRAIN_SIZE + 1:]

        np.random.shuffle(self.orange_set)  # put the points in a random order

        # take the first part of the shuffled list for traning and last part for test
        self.orange_training_set = self.orange_set[0:self.ORANGE_TRAIN_SIZE]
        self.orange_test_set = self.orange_set[self.ORANGE_TRAIN_SIZE + 1:]

        self.model = linear_model.LinearRegression()
        self.model.fit(
            np.concatenate((self.blue_training_set, self.orange_training_set)),
            np.concatenate((np.zeros(self.BLUE_TRAIN_SIZE),
                            np.ones(self.ORANGE_TRAIN_SIZE))),
        )

        # Determine the x1 and x2 intercepts in order to plot the decision
        # boundary which is 1/2 = Ax+b.
        self.x1 = (.5 - self.model.intercept_) / self.model.coef_[1]
        self.x2 = (.5 - self.model.intercept_) / self.model.coef_[0]

    def plot_results(self, axis=None, xlimits=None, ylimits=None):
        # if an axis object is given then add the plots to it and return it otherwise
        # create a figure and return that
        if not axis:
            fig1 = plt.figure()
            ax = fig1.add_subplot(111)
        else:
            fig1 = None
            ax = axis

        ax.scatter(
            x=self.blue_training_set[:, 0],
            y=self.blue_training_set[:, 1],
            color="blue")
        ax.scatter(
            x=self.orange_training_set[:, 0],
            y=self.orange_training_set[:, 1],
            color="orange",
        )
        ax.scatter(
            x=self.blue_test_set[:, 0],
            y=self.blue_test_set[:, 1],
            color="blue",
            marker="+",
        )
        ax.scatter(
            x=self.orange_test_set[:, 0],
            y=self.orange_test_set[:, 1],
            color="orange",
            marker="+",
        )
        ax.plot([self.x1, 0], [0, self.x2],
                color="purple")  # plot intercepts with line connecting

        if xlimits:
            ax.set_xlim(xlimits)
        if ylimits:
            ax.set_ylim(ylimits)

        if fig1:
            return fig1
        else:
            return ax

    def get_errors(self):
        self.errors = []
        for index, prediction in enumerate(
                self.model.predict(
                    np.concatenate((self.blue_training_set,
                                    self.blue_test_set)))):
            if prediction > .5:
                self.errors.append(
                    np.concatenate((self.blue_training_set,
                                    self.blue_test_set))[index])

        for index, prediction in enumerate(
                self.model.predict(
                    np.concatenate((self.orange_training_set,
                                    self.orange_test_set)))):
            if prediction < .5:
                self.errors.append(
                    np.concatenate((self.orange_training_set,
                                    self.orange_test_set))[index])
        return self.errors
