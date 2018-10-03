"""
resturant_data.py
=================
Data exploration for my interview with The Filter.

"""
from numpy import array, delete
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import mean_absolute_error

resturants = pd.read_csv('./data/restaurants.csv')
resturants.isnull().values.any()


def explore():
    for column in resturants.columns:
        print(column, ' : ', resturants[column].unique(), '\n')


resturants = resturants.astype('category')
resturants.age = resturants.age.cat.reorder_categories(
    ['under_19', '20_34', '35_49', '50_64', '65_and_over'], ordered=True)
resturants.budget = resturants.budget.cat.reorder_categories(
    ['under_20', '20_to_30', '30_to_50', 'over_50'], ordered=True)
resturants.price = resturants.price.cat.reorder_categories(
    ['under_20', '20_to_30', '30_to_50', 'over_50'], ordered=True)
resturants.rating = resturants.rating.cat.reorder_categories(
    ['dislike', 'satisfactory', 'very good', 'excellent'], ordered=True)
resturants.cuisine_type.cat.rename_categories(
    [cat[0:11] for cat in resturants.cuisine_type.cat.categories])
resturants['stars'] = resturants.rating.cat.codes
resturants['over_budget'] = pd.Categorical(
    resturants.budget < resturants.price)

data_size = len(resturants)


def explore_visual():
    resturants.describe()
    len(resturants)

    fig, axes = plt.subplots(2, 3)
    for i, column in enumerate(resturants.columns):
        sns.countplot(resturants[column], ax=axes[i % 2, i % 3])

    axis = sns.countplot(resturants.cuisine_type)
    axis.set_xticklabels(axis.get_xticklabels(), rotation=90)


def percentplot(data, **kwargs):
    ax = sns.barplot(
        x=data,
        y=data,
        orient='v',
        ci=None,
        estimator=lambda x: len(x) * 100.0 / len(data),
        alpha=.5,
        **kwargs)

    # ratio = '{:d} of {:d}'.format(len(data), data_size)
    float_percent = len(data) * 100.0 / data_size
    if ax.texts:
        float_percent = len(data) * 100.0 / data_size
        for text in ax.texts:
            float_percent += float(text.get_text()[:-1])
        ax.texts = []

    percent = '{:.2g}%'.format(float_percent)

    ax.text(.5, .965, percent, transform=ax.transAxes)
    return ax


def facetplot(facet):
    # price vs age
    # age vs rating
    for col in resturants.columns:
        if col != facet:
            g = sns.FacetGrid(resturants, col=facet)
            g.map(percentplot, col)


def bigfacet(data, facet, col, row=None, hue=None):
    if hue and row:
        g = sns.FacetGrid(data, row=row, hue=hue, col=col, palette='Set1')
        g.map(percentplot, facet)
        g.set_titles('{row_name:.1s}|{col_name:.8s}')
        g.add_legend()
    elif row:
        g = sns.FacetGrid(data, row=row, col=col, palette='Set1')
        g.map(percentplot, facet)
        g.set_titles('{row_name:.1s}|{col_name:.8s}')
    elif hue:
        g = sns.FacetGrid(data, hue=hue, col=col, palette='Set1')
        g.map(percentplot, facet)
        g.set_titles('{col_name:.8s}')
        g.add_legend()

    x_tick_labels = []
    for cat in data[facet].cat.categories:
        x_tick_labels.append(cat[0])

    g.set_xticklabels(x_tick_labels)


def chi_square(row, col, data, lambda_=1):
    row_vars = data[row].cat.categories
    col_vars = data[col].cat.categories
    observations = []

    for var1 in row_vars:
        col_obs = []
        for var2 in col_vars:
            col_obs.append(
                len(
                    data.query('{row} == @var1'.format(row=row)).query(
                        '{col} == @var2'.format(col=col))))
        observations.append(col_obs)

    obs = array(observations)
    zeros, dropped = [], 0

    for index, elem in enumerate(obs.sum(axis=0)):
        if elem == 0:
            zeros.append(index)
            dropped += 1
    obs = delete(obs, zeros, axis=1)
    return chi2_contingency(obs, lambda_=lambda_)


def chi2_against_ratings(col, lambda_=1):
    print('-------{col_name}-------'.format(col_name=col))
    for cuisine in resturants.cuisine_type.cat.categories:
        pval = chi_square(
            col,
            'rating',
            resturants[resturants.cuisine_type == cuisine],
            lambda_=0)[1:3]
        print('{cuisine} : {pval}'.format(cuisine=cuisine, pval=pval))


features = ['gender', 'age', 'cuisine_type', 'over_budget']


def predict_with_tree(features, label, tree_type='regressor', **kwargs):
    feats = []
    for col in features:
        feats.append(resturants[col].cat.codes.values)
    X = array(feats)
    X = X.T
    y = resturants[label].cat.codes.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.40)
    if tree_type == 'regressor':
        clf = tree.DecisionTreeRegressor(**kwargs)
    elif tree_type == 'classifier':
        clf = tree.DecisionTreeClassifier(**kwargs)
    else:
        raise ValueError('tree_type must be one of [regressor, classifier]')
    clf.fit(X_train, y_train)
    return clf, X_test, y_test


def test_model(model, *args, **kwargs):
    clf, X_test, y_test = model(*args, **kwargs)
    print(mean_absolute_error(y_test, clf.predict(X_test)))
