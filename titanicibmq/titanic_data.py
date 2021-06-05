# -*- coding: utf-8 -*-
#
# Written by Adel Sohbi, https://github.com/adelshb
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import pandas as pd
import numpy as np
import random as rnd

def parse_data_train_vqc(data, split_ratio=0.2):

	data.sample(frac=1)

	train_df = data.iloc[int(data.shape[0]*split_ratio):,:]
	test_df = data.iloc[:int(data.shape[0]*split_ratio),:]

	X_train = train_df.drop("Survived", axis=1).values
	y_train = train_df["Survived"].values

	X_test = test_df.drop("Survived", axis=1).values
	y_test = test_df["Survived"].values

	return X_train, y_train, X_test, y_test

def titanic(datapath="data/"):
	"""
	This code is a modified version of the following notebook: https://www.kaggle.com/startupsci/titanic-data-science-solutions
	"""
	train_df = pd.read_csv(datapath + 'train.csv')
	test_df = pd.read_csv(datapath + 'test.csv')
	combine = [train_df, test_df]

	train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
	test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
	combine = [train_df, test_df]

	for dataset in combine:
	    dataset['Title'] = dataset.Name.str.extract(r' ([A-Za-z]+)\.', expand=False)

	pd.crosstab(train_df['Title'], train_df['Sex'])

	for dataset in combine:
	    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
	 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

	    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
	    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
	    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

	train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

	title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
	for dataset in combine:
	    dataset['Title'] = dataset['Title'].map(title_mapping)
	    dataset['Title'] = dataset['Title'].fillna(0)

	train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
	test_df = test_df.drop(['Name'], axis=1)
	combine = [train_df, test_df]

	for dataset in combine:
	    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

	guess_ages = np.zeros((2,3))
	for dataset in combine:
	    for i in range(0, 2):
	        for j in range(0, 3):
	            guess_df = dataset[(dataset['Sex'] == i) & \
	                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

	            age_guess = guess_df.median()

	            # Convert random age float to nearest .5 age
	            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

	    for i in range(0, 2):
	        for j in range(0, 3):
	            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
	                    'Age'] = guess_ages[i,j]

	    dataset['Age'] = dataset['Age'].astype(int)

	train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
	train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

	for dataset in combine:
	    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
	    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
	    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
	    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
	    dataset.loc[ dataset['Age'] > 64, 'Age']

	train_df = train_df.drop(['AgeBand'], axis=1)
	combine = [train_df, test_df]

	for dataset in combine:
	    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

	train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)

	for dataset in combine:
	    dataset['IsAlone'] = 0
	    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

	train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

	train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
	test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
	combine = [train_df, test_df]

	for dataset in combine:
	    dataset['Age*Class'] = dataset.Age * dataset.Pclass

	freq_port = train_df.Embarked.dropna().mode()[0]
	for dataset in combine:
	    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

	train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)

	for dataset in combine:
	    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

	test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

	train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
	train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)

	for dataset in combine:
	    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
	    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
	    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
	    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
	    dataset['Fare'] = dataset['Fare'].astype(int)

	train_df = train_df.drop(['FareBand'], axis=1)

	return train_df, test_df
