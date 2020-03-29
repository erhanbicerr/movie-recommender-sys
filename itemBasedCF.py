import pandas as pd
import numpy as np

r_cols =['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('D:\\MLCourse\\ml-100k\\u.data', sep='\t', names=r_cols, usecols=range(3), encoding="ISO-8859-1") 

m_cols = ['movie_id', 'title']
movies = pd.read_csv('D:\\MLCourse\\ml-100k\\u.item', sep='|', names=m_cols, usecols=range(2), encoding="ISO-8859-1") 

ratings = pd.merge(movies, ratings)

#print(ratings.head())

userRatings =  ratings.pivot_table(index=['user_id'], columns=['title'], values='rating')
#print(userRatings.head())

corrMatrix = userRatings.corr()
#print(corrMatrix.head())

corrMatrix = userRatings.corr(method='pearson', min_periods = 100)
#print(corrMatrix.head())


data = {}
x = int(input("How many movie ratings you have ?"))

for i in range (0,x):
    movie=input("Which Movie ? ")
    rating = float(input("Rating? (0-5)"))
    data[movie]=rating
    
    
myRatings = pd.Series(data)
print("Your Ratings ;")
print(myRatings)

simCandidates = pd.Series()
for i in range(0, len(myRatings.index)):
    print (" Adding sims for " + myRatings.index[i] + "...")
    sims = corrMatrix[myRatings.index[i]].dropna()
    sims = sims.map(lambda x: x * myRatings[i])
    simCandidates = simCandidates.append(sims)
    
    
#simCandidates.sort_values(inplace = True, ascending = False)
#print(simCandidates.head(5))

simCandidates = simCandidates.groupby(simCandidates.index).sum()
simCandidates.sort_values(inplace = True, ascending = False)
#print(simCandidates.head(10))
filteredSims = simCandidates.drop(myRatings.index)
print("YOU MAY LOVE THESE MOVIES TOO ;")
print(filteredSims.head(15))
