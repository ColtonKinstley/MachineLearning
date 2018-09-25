"""
olympic-lasso.py
================
Predicitve models for kaggle olympic data.

"""
import pandas as pd
from sklearn import neighbors

n_neighbors = 3
athlete_data = pd.read_csv('./data/olympic-data/athlete_events.csv')

top_five_males = athlete_data.query('Sex == "M"').loc[athlete_data.Sport.isin(athlete_data.loc[
    athlete_data.Sex == 'M'].Sport.value_counts().head(5).index)]
top_five_males['sport_id'] = top_five_males.Sport.astype('category').cat.codes
top_five_males = top_five_males.loc[:, ['Weight', 'Height', 'sport_id']]

top_five_train = top_five_males.sample(frac=.9)
top_five_test = top_five_males.drop(top_five_train.index)

neighs = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
neighs.fit()
