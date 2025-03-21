import os
import numpy as np
"""This is just a random python file for test purpose!"""

# Given path
path = r"D:\digital library\Business,finance and economics\The_Role_of_Public_Relations_in_Branding.pdf"

# Extract the last part with .pdf extension
filename = os.path.basename(path)

print(filename)  # Output: The_Role_of_Public_Relations_in_Branding.pdf


# completed 
"""
1. The_Role_of_Public_Relations_in_Branding.pdf
2. 19_018_fundamentals_of_taxation_final_web.pdf
3. Rich Dad Poor Dad.pdf
4. artificial_intelligence_in_finance_-_turing_report_0.pdf
5. the-art-of-seo-by-eric-and-jessie.pdf

"""


class k_means:

    def __init__(self,clusters):
        self.k = clusters

    def fit(self,X_train):

        # Normalise the datapoints
        X_train = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)

        # randomly select k data points from the training data as the initial clusters(without replacement)
        self.centroids = X_train[np.random.choice(X_train.shape[0], size=self.k, replace=False)]


        # Assign clusters for every point in X_train
        # euclidean_distances = np.linalg.norm(X_train[:, np.newaxis, :] - self.centroids[np.newaxis, :, :], axis=2)
        
        cosine_similarity = np.dot(X_train, self.centroids.T)

        # Find the index of the centroid with the highest similarity for each data point
        closest_points_indices = np.argmax(cosine_similarity, axis=1)

        # Fetch the closest centroids for each point
        closest_centroids = self.centroids[closest_points_indices]
        
        d = {}
        for i, j in zip(closest_centroids, X_train):
            i_tuple = tuple(i)  # Convert to regular Python float
            if i_tuple in d:
                d[i_tuple].append(j)
            else:
                d[i_tuple] = [j]
        
        
        # calculate the centroid of the generated k clusters
        for i in d.values():
            new_centroids = self.find_centroid(i)


        # repeat the steps until the previous set of centroid is same as the current set of centroids

        return d,self.centroids,closest_centroids
    

data = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8],
    [9, 10]
])

clf = k_means(3)
x,y,z= clf.fit(data)
print(x)
print(len(x))

print()
print(y)
print(f"\n{list(x.keys())}")
# print(tuple(y))
# cl,e,cs = clf.fit(data)
# print(cl)
# print()
# print(e)
# print()
# print(cs)