import pandas
from sklearn.tree import DecisionTreeClassifier

music_data = pandas.read_csv('music.csv')
X = music_data.drop(columns=['genre'])
y = music_data['genre']

model = DecisionTreeClassifier()
model.fit(X, y)
predictions = model.predict([ [21,1],[22,0] ])

print("Done")