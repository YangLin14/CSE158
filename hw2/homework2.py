#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import defaultdict
from sklearn import linear_model
import numpy
import math


# In[ ]:





# In[ ]:


def feat(d, catID, maxLength, includeCat = True, includeReview = True, includeLength = True):
    feat = []
    if includeCat:
        style = d.get('beer/style', None)
        if style and style in catID:
            feat.extend([1 if i == catID[style] else 0 for i in range(len(catID))])
        else:
            feat.extend([0] * len(catID))
    if includeReview:
        # For Q2
        ratings = [
            d.get('review/aroma', 0),
            d.get('review/appearance', 0),
            d.get('review/palate', 0),
            d.get('review/taste', 0),
            d.get('review/overall', 0)
        ]
        feat.extend(ratings)
    if includeLength:
        # For Q2
        feat.append(len(d['review/text']) / maxLength if maxLength > 0 else 0)
    return feat + [1]


# In[ ]:


def pipeline(reg, catID, dataTrain, dataValid, dataTest, includeCat=True, includeReview=True, includeLength=True):
    mod = linear_model.LogisticRegression(C=reg, class_weight='balanced')

    maxLength = max([len(d['review/text']) for d in dataTrain]) if dataTrain else 1
    
    Xtrain = [feat(d, catID, maxLength, includeCat, includeReview, includeLength) for d in dataTrain]
    yTrain = [d['beer/ABV'] > 7 for d in dataTrain]
    
    mod.fit(Xtrain, yTrain)
    
    Xvalid = [feat(d, catID, maxLength, includeCat, includeReview, includeLength) for d in dataValid]
    yValid = [d['beer/ABV'] > 7 for d in dataValid]
    
    Xtest = [feat(d, catID, maxLength, includeCat, includeReview, includeLength) for d in dataTest]
    yTest = [d['beer/ABV'] > 7 for d in dataTest]
    
    def calculate_ber(X, y):
        preds = mod.predict(X)
        TP = sum(p and l for p, l in zip(preds, y))
        TN = sum(not p and not l for p, l in zip(preds, y))
        FP = sum(p and not l for p, l in zip(preds, y))
        FN = sum(not p and l for p, l in zip(preds, y))
        
        if (FP + TN) == 0 or (FN + TP) == 0:
            return 0
        
        FPR = FP / (FP + TN)
        FNR = FN / (FN + TP)
        return 0.5 * (FPR + FNR)

    vBER = calculate_ber(Xvalid, yValid)
    tBER = calculate_ber(Xtest, yTest)

    return mod, vBER, tBER


# In[ ]:





# In[2]:


### Question 1


# In[ ]:


def Q1(catID, dataTrain, dataValid, dataTest):
    # No need to modify this if you've implemented the functions above
    mod, validBER, testBER = pipeline(10, catID, dataTrain, dataValid, dataTest, True, False, False)
    return mod, validBER, testBER


# In[ ]:





# In[3]:


### Question 2


# In[ ]:


def Q2(catID, dataTrain, dataValid, dataTest):
    mod, validBER, testBER = pipeline(10, catID, dataTrain, dataValid, dataTest, True, True, True)
    return mod, validBER, testBER


# In[ ]:





# In[ ]:


### Question 3


# In[ ]:


def Q3(catID, dataTrain, dataValid, dataTest):
    best_mod = None
    best_vBER = float('inf')
    best_tBER = float('inf')

    for c in [0.001, 0.01, 0.1, 1, 10]:
        mod, vBER, tBER = pipeline(c, catID, dataTrain, dataValid, dataTest, True, True, True)
        if vBER < best_vBER:
            best_vBER = vBER
            best_tBER = tBER
            best_mod = mod
            
    return best_mod, best_vBER, best_tBER


# In[ ]:





# In[4]:


### Question 4


# In[11]:


def Q4(C, catID, dataTrain, dataValid, dataTest):
    mod, validBER, testBER_noCat = pipeline(C, catID, dataTrain, dataValid, dataTest, False, True, True)
    mod, validBER, testBER_noReview = pipeline(C, catID, dataTrain, dataValid, dataTest, True, False, True)
    mod, validBER, testBER_noLength = pipeline(C, catID, dataTrain, dataValid, dataTest, True, True, False)
    return testBER_noCat, testBER_noReview, testBER_noLength


# In[ ]:





# In[ ]:


### Question 5


# In[ ]:


def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0:
        return 0
    return numer / denom


# In[ ]:


def mostSimilar(i, N, usersPerItem):
    similarities = []
    users_i = usersPerItem.get(i, set())
    for j in usersPerItem:
        if i == j:
            continue
        users_j = usersPerItem[j]
        sim = Jaccard(users_i, users_j)
        similarities.append((sim, j))
    
    similarities.sort(key=lambda x: x[0], reverse=True)
    
    return similarities[:N]


# In[ ]:





# In[7]:


### Question 6


# In[8]:


def MSE(y, ypred):
    return numpy.mean((numpy.array(y) - numpy.array(ypred)) ** 2)


# In[ ]:


def getMeanRating(dataTrain):
    if not dataTrain:
        return 0
    return numpy.mean([d['star_rating'] for d in dataTrain])

def getUserAverages(itemsPerUser, ratingDict):
    userAverages = {}
    for u in itemsPerUser:
        ratings = [ratingDict[(u, i)] for i in itemsPerUser[u]]
        if ratings:
            userAverages[u] = numpy.mean(ratings)
    return userAverages

def getItemAverages(usersPerItem, ratingDict):
    itemAverages = {}
    for i in usersPerItem:
        ratings = [ratingDict[(u, i)] for u in usersPerItem[i]]
        if ratings:
            itemAverages[i] = numpy.mean(ratings)
    return itemAverages


# In[ ]:





# In[9]:


def predictRating(user,item,ratingMean,reviewsPerUser,usersPerItem,itemsPerUser,userAverages,itemAverages):
    ratings = []
    similarities = []
    
    user_ratings_map = {d['product_id']: d['star_rating'] for d in reviewsPerUser.get(user, [])}
    
    # Neighbor items rated by the user
    for j in itemsPerUser.get(user, []):
        if item == j:
            continue
        
        sim = Jaccard(usersPerItem.get(item, set()), usersPerItem.get(j, set()))
        
        if sim > 0:
            # Use user_ratings_map instead of ratingDict
            rating_uj = user_ratings_map.get(j, ratingMean)
            ratings.append(rating_uj - itemAverages.get(j, ratingMean))
            similarities.append(sim)
            
    if not ratings:
        return itemAverages.get(item, ratingMean)
        
    numerator = sum(r * s for r, s in zip(ratings, similarities))
    denominator = sum(similarities)
    
    if denominator == 0:
        return itemAverages.get(item, ratingMean)
        
    return itemAverages.get(item, ratingMean) + numerator / denominator


# In[ ]:





# In[10]:


### Question 7


# In[ ]:

def predictRatingQ7(user, item, ratingMean, reviewsPerUser, usersPerItem, itemsPerUser, userAverages, itemAverages):
    prediction = ratingMean
    
    user_count = len(itemsPerUser.get(user, [])) if user in itemsPerUser else 0
    item_count = len(usersPerItem.get(item, [])) if item in usersPerItem else 0
    
    if user_count >= 10:
        user_avg = userAverages.get(user, ratingMean)
        user_bias = user_avg - ratingMean
        prediction += 0.3 * user_bias
    
    if item_count >= 20:
        item_avg = itemAverages.get(item, ratingMean)
        item_bias = item_avg - ratingMean
        prediction += 0.2 * item_bias
    
    return max(1.0, min(5.0, prediction))


# In[ ]:




