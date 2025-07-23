import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Recommender:
    def __init__(self, path='movies.csv'):
        self.df = pd.read_csv(path)
        self.df.dropna(inplace=True)
        self.vectorizer = CountVectorizer(tokenizer=lambda x: x.split('|'))
        self.count_matrix = self.vectorizer.fit_transform(self.df['genres'])
        self.similarity = cosine_similarity(self.count_matrix)

    def recommend(self, title, top_n=3):
        if title not in self.df['title'].values:
            return []

        index = self.df[self.df['title'] == title].index[0]
        similarity_scores = list(enumerate(self.similarity[index]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        recommended_indices = [i[0] for i in similarity_scores[1:top_n+1]]
        return self.df.iloc[recommended_indices]['title'].tolist()
