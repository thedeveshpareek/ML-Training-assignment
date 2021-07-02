import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv('movie_metadata.csv')

df.drop(["director_name","actor_2_name","genres","movie_title","actor_1_name","actor_3_name","language","country","content_rating"],axis=1, inplace = True)
df.drop(["plot_keywords","movie_imdb_link"],axis=1, inplace = True)

df.dropna(subset = ["color"], axis=0, inplace=True)
df.dropna(how="any",axis=0,inplace = True)


target = df["imdb_score"]
data = df.drop(["imdb_score"],axis=1)

labelencoder = LabelEncoder()
data["new_color"] = labelencoder.fit_transform(data["color"])

data.drop(["color"],axis=1,inplace=True)
from sklearn.preprocessing import scale
scaled_data = scale(data)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(scaled_data, target, test_size = 0.25)

from sklearn.linear_model import LinearRegression

#training the model (fitting data to the model)
model = LinearRegression()
model.fit(x_train,y_train)

y = model.predict(x_test)

predictions = pd.DataFrame({"Actual":y_test, "Predicted":y})

# Evaluating the model
from sklearn.preprocessing import MinMaxScaler
mmscaler = MinMaxScaler()
scaled_targets = mmscaler.fit_transform(predictions)

scaled_targets = pd.DataFrame(scaled_targets,columns = ["Actual","Predicted"])


from sklearn import tree
dtree = tree.DecisionTreeRegressor(max_depth = 7)
dtree.fit(x_train,y_train)

y = dtree.predict(x_test)

predictions = pd.DataFrame({"Actual":y_test, "Predicted":y})

l = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

def evaluateModel(d):
    dtree = tree.DecisionTreeRegressor(max_depth = d)
    dtree.fit(x_train,y_train)
    return dtree.score(x_test,y_test)

scores = []
for d in l:
    scores.append(evaluateModel(d))

import matplotlib.pyplot as plt
# %matplotlib inline

# plt.plot(np.array(l),np.array(scores))
# plt.show()
clf=[dtree,model]
import pickle 
pickle.dump(model,open("model.pkl","wb"))
pickle.dump(dtree,open("Dtree_model.pkl","wb"))


