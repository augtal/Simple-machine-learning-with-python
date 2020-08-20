import pandas
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn import tree

music_data = pandas.read_csv('music.csv')
X = music_data.drop(columns=['genre'])
y = music_data['genre']
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

tree.export_graphviz(   model, out_file='music-recommender.dot', feature_names=['age', 'gender'], 
                        class_names=sorted(y.unique()), label='all', rounded=True, filled=True)

predictions = model.predict(X_test)

acc_Score = accuracy_score(y_test, predictions)
print(acc_Score)
print("Done")


"""
joblib.dump(model, 'music-recomender.joblib')
model = joblib.load('music-recomender.joblib')
"""