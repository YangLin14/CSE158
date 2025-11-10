#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from collections import defaultdict
import math
import scipy.optimize
import numpy
import string
from sklearn import linear_model
import random


# In[ ]:


def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)


# In[ ]:


def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r


# In[ ]:


def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom > 0:
        return numer/denom
    return 0


# In[ ]:


##################################################
# Rating prediction                              #
##################################################


# In[ ]:


def getGlobalAverage(trainRatings):
    # Return the average rating in the training set
    return sum(trainRatings) / len(trainRatings)


# In[ ]:


def trivialValidMSE(ratingsValid, globalAverage):
    # Compute and return the MSE of a trivial model that always returns the global mean computed above
    squared_error = 0
    for user, item, rating in ratingsValid:
        squared_error += (rating - globalAverage) ** 2
    return squared_error / len(ratingsValid)


# In[ ]:


def alphaUpdate(ratingsTrain, alpha, betaU, betaI, lamb):
    # Update equation for alpha
    total_error = 0
    for user, item, rating in ratingsTrain:
        bu = betaU.get(user, 0)
        bi = betaI.get(item, 0)
        total_error += rating - ( bu + bi)
    newAlpha = total_error / len(ratingsTrain)
    return newAlpha


# In[ ]:


def betaUUpdate(ratingsPerUser, alpha, betaU, betaI, lamb):
    # Update equation for betaU
    newBetaU = {}
    for user in ratingsPerUser:
        total_error = 0
        Iu = ratingsPerUser[user]
        
        for (b, r) in Iu:
            bi = betaI.get(b, 0)
            
            total_error += (r - (alpha + bi))
        
        newBetaU[user] = total_error / (lamb + len(Iu))
    return newBetaU


# In[ ]:


def betaIUpdate(ratingsPerItem, alpha, betaU, betaI, lamb):
    # Update equation for betaI
    newBetaI = {}
    for b in ratingsPerItem:
        total_error = 0
        Ui = ratingsPerItem[b]
        
        for (u, r) in Ui:
            bu = betaU.get(u, 0)
            
            total_error += (r - (alpha + bu))
        
        newBetaI[b] = total_error / (lamb + len(Ui))
    return newBetaI


# In[ ]:


def msePlusReg(ratingsTrain, alpha, betaU, betaI, lamb):
    # Compute the MSE and the mse+regularization term
    mse = 0
    regularizer = 0

    for user, item, rating in ratingsTrain:
        bu = betaU.get(user, 0)
        bi = betaI.get(item, 0)
        mse += (rating - (alpha + bu + bi)) ** 2
    mse /= len(ratingsTrain)

    for user in betaU:
        regularizer += betaU[user] ** 2
    for b in betaI:
        regularizer += betaI[b] ** 2
    return mse, mse + lamb*regularizer


# In[ ]:


def validMSE(ratingsValid, alpha, betaU, betaI):
    # Compute the MSE on the validation set
    squared_error = 0
    for user, item, rating in ratingsValid:
        bu = betaU.get(user, 0)
        bi = betaI.get(item, 0)
        squared_error += (rating - (alpha + bu + bi)) ** 2
    validMSE = squared_error / len(ratingsValid)
    return validMSE


# In[ ]:


def goodModel(ratingsTrain, ratingsPerUser, ratingsPerItem, alpha, betaU, betaI):
    # Improve upon your model from the previous question (e.g. by running multiple iterations)
    lamb = 1.0
    N_iterations = 5

    for i in range(N_iterations):
        alpha = alphaUpdate(ratingsTrain, alpha, betaU, betaI, lamb)
        betaU = betaUUpdate(ratingsPerUser, alpha, betaU, betaI, lamb)
        betaI = betaIUpdate(ratingsPerItem, alpha, betaU, betaI, lamb)
    return alpha, betaU, betaI


# In[ ]:


def writePredictionsRating(alpha, betaU, betaI):
    # Write your predictions to a file that you can submit
    predictions = open("predictions_Rating.csv", 'w')
    for l in open("pairs_Rating.csv"):
        if l.startswith("userID"):
            predictions.write(l)
            continue
        u,b = l.strip().split(',')
        bu = 0
        bi = 0
        if u in betaU:
            bu = betaU[u]
        if b in betaI:
            bi = betaI[b]
        _ = predictions.write(u + ',' + b + ',' + str(alpha + bu + bi) + '\n')

    predictions.close()


# In[ ]:


##################################################
# Read prediction                                #
##################################################


# In[ ]:


def generateValidation(allRatings, ratingsValid):
    # Using ratingsValid, generate two sets:
    # readValid: set of (u,b) pairs in the validation set
    # notRead: set of (u,b') pairs, containing one negative (not read) for each row (u) in readValid  
    # Both should have the same size as ratingsValid
    allBooks = set()
    allRead = set()
    for (user, book, rating) in allRatings:
        allRead.add((user, book))
        allBooks.add(book)
    
    readValid = set()
    for (user, book, rating) in ratingsValid:
        readValid.add((user, book))

    notRead = set()
    allBookList = list(allBooks)

    for (user, book) in readValid:
        while True:
            random_book = random.choice(allBookList)
            if (user, random_book) not in allRead:
                notRead.add((user, random_book))
                break

    return readValid, notRead

# In[ ]:


def baseLineStrategy(mostPopular, totalRead):
    return1 = set()

    # Compute the set of items for which we should return "True"
    # This is the same strategy implemented in the baseline code for Assignment 1
    return1 = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        return1.add(i)
        if count > totalRead/2: break

    return return1


# In[ ]:


def improvedStrategy(mostPopular, totalRead):
    return1 = set()

    # Same as above function, just find an item set that'll have higher accuracy

    threshold = totalRead * 0.60 
    
    count = 0
    for ic, i in mostPopular:
        count += ic
        return1.add(i)
        if count > threshold: break

    return return1


# In[ ]:


def evaluateStrategy(return1, readValid, notRead):

    # Compute the accuracy of a strategy which just returns "true" for a set of items (those in return1)
    # readValid: instances with positive label
    # notRead: instances with negative label

    correct = 0
    
    for user, book in readValid:
        if book in return1:
            correct += 1

    for user, book in notRead:
        if book not in return1:
            correct += 1

    acc = correct / (len(readValid) + len(notRead))
    return acc


# In[ ]:


def jaccardThresh(u,b,ratingsPerItem,ratingsPerUser):
    
    # Compute the similarity of the query item (b) compared to the most similar item in the user's history
    # Return true if the similarity is high or the item is popular

    users_who_read_b = set([user for (user, rating) in ratingsPerItem.get(b, [])])
    
    books_read_by_u = ratingsPerUser.get(u, [])
    
    maxSim = 0
    
    for (b_prime, r) in books_read_by_u:
        if b == b_prime: continue
            
        users_who_read_b_prime = set([user for (user, rating) in ratingsPerItem.get(b_prime, [])])
        
        numer = len(users_who_read_b.intersection(users_who_read_b_prime))
        denom = len(users_who_read_b.union(users_who_read_b_prime))
        
        sim = 0
        if denom > 0:
            sim = numer / denom
            
        if sim > maxSim:
            maxSim = sim
    
    if maxSim > 0.013 or len(ratingsPerItem[b]) > 40: # Keep these thresholds as-is
        return 1
    return 0


# In[ ]:


def writePredictionsRead(ratingsPerItem, ratingsPerUser):
    predictions = open("predictions_Read.csv", 'w')
    for l in open("pairs_Read.csv"):
        if l.startswith("userID"):
            predictions.write(l)
            continue
        u,b = l.strip().split(',')
        pred = jaccardThresh(u,b,ratingsPerItem,ratingsPerUser)
        _ = predictions.write(u + ',' + b + ',' + str(pred) + '\n')

    predictions.close()


# In[ ]:


##################################################
# Category prediction                            #
##################################################


# In[ ]:


def featureCat(datum, words, wordId, wordSet):
    feat = [0]*len(words)

    # Compute features counting instance of each word in "words"
    # after converting to lower case and removing punctuation
    punctuation = set(string.punctuation)
    r = ''.join([c for c in datum['review_text'].lower() if c not in punctuation])

    for w in r.split():
        if w in wordSet:
            idx = wordId[w]
            feat[idx] += 1
    
    feat.append(1) # offset (put at the end)
    return feat


# In[ ]:


def betterFeatures(data):
    
    # Produce better features than those from the above question
    # Return matrix (each row is the feature vector for one entry in the dataset)

    wordCount = defaultdict(int)
    punctuation = set(string.punctuation)
    
    for d in data:
        r = ''.join([c for c in d['review_text'].lower() if not c in punctuation])
        for w in r.split():
            wordCount[w] += 1
            
    counts = [(wordCount[w], w) for w in wordCount]
    counts.sort()
    counts.reverse()
    
    NW = 2000 
    words = [x[1] for x in counts[:NW]]
    
    wordId = dict(zip(words, range(len(words))))
    wordSet = set(words)
    
    X = []
    for d in data:
        feat = [0] * len(words)
        r = ''.join([c for c in d['review_text'].lower() if not c in punctuation])
        
        for w in r.split():
            if w in wordSet:
                feat[wordId[w]] += 1
                
        feat.append(1)
        X.append(feat)
    return X


# In[ ]:


def runOnTest(data_test, mod):
    Xtest = [featureCat(d) for d in data_test]
    pred_test = mod.predict(Xtest)


# In[ ]:


def writePredictionsCategory(pred_test):
    predictions = open("predictions_Category.csv", 'w')
    pos = 0

    for l in open("../datasets/pairs_Category.csv"):
        if l.startswith("userID"):
            predictions.write(l)
            continue
        u,b = l.strip().split(',')
        _ = predictions.write(u + ',' + b + ',' + str(pred_test[pos]) + '\n')
        pos += 1

    predictions.close()


# In[ ]:




