import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv('movie_metadata.csv')
df['title_year'].fillna(0,inplace=True)
df['title_year'] = df['title_year'].apply(np.int64)


#movie recommendation
df2 = df.sort_values('imdb_score', ascending=False)

dataset= df[['director_name','actor_2_name','genres','title_year','actor_1_name','movie_title','actor_3_name']]
## clean genres--- remove | between generes
dataset['genres'] = dataset['genres'].apply(lambda a: str(a).replace('|', ' '))
dataset['movie_title'][0]
dataset['movie_title'] = dataset['movie_title'].apply(lambda a:a[:-1])
## combined features on which we will calculate cosine similarity

dataset['director_genre_actors'] = dataset['director_name']+' '+dataset['actor_1_name']+' '+' '+dataset['actor_2_name']+' '+dataset['actor_3_name']+' '+dataset['genres']

dataset.fillna('', inplace=True)

print(dataset.isnull().sum())


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vec = CountVectorizer()
vec_matrix = vec.fit_transform(dataset['director_genre_actors'])

similarity = cosine_similarity(vec_matrix)

class movie:
    def recommend_movie(movie):
        if movie not in dataset['movie_title'].unique():
            message='SorryThe movie you requested is not in our database. Please check the spelling or try with some other movies'
            df0=pd.DataFrame({'Movies Recommended':[message],'Year':[0000]})
            return df0
        else:
            i = dataset.loc[dataset['movie_title']==movie].index[0]
            lst = list(enumerate(similarity[i]))
            lst = sorted(lst, key = lambda x:x[1] ,reverse=True)
            lst = lst[1:11] # excluding first item since it is the requested movie itself
            l = []
            year=[]
            for i in range(len(lst)):
                a = lst[i][0]
                l.append(dataset['movie_title'][a])
                year.append(dataset['title_year'][a])
                
            # plt.figure(figsize=(10,5))
            # plt.bar(l, [i[1] for i in lst])
            # plt.xticks(rotation=90)
            # plt.xlabel('Movies similar to → '+movie, fontsize=12,fontweight="bold")
            # plt.ylabel('cosine scores', fontsize=12,fontweight="bold")
            # plt.show()
            df2 = pd.DataFrame({'Movies Recommended':l, 'Year':year})
            df2.drop_duplicates
            
    
            return df2
