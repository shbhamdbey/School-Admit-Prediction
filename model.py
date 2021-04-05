import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
data = pd.read_csv('https://stats.idre.ucla.edu/stat/data/binary.csv')


corr = data.corr()


ax = sns.heatmap(corr, linewidth=0.5, annot=True)
plt.show()

X = data[['gre','gpa','rank']]
y = data['admit']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


model = LogisticRegression()

model.fit(X_train, y_train)

model.predict(X_test)

model.score(X_test,y_test)

with open('model_pickle', 'wb') as f:
    pickle.dump(model,f)

