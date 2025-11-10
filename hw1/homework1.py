#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import defaultdict
from sklearn import linear_model
import numpy
import math


# In[2]:


### Question 1


# In[3]:


def getMaxLen(dataset):
    # Find the longest review (number of characters)
    maxLen = max([len(d['review_text']) for d in dataset if 'review_text' in d])
    if not dataset:
        return 1
    # Use a generator expression for memory efficiency
    maxLen = max((len(d['review_text']) for d in dataset if 'review_text' in d), default=0)
    # Avoid division by zero if all reviews are empty or missing
    return maxLen


# In[4]:


def featureQ1(datum, maxLen):
    # Feature vector for one data point
    review_text = datum['review_text']

    # Bewteen 0 and 1
    scaled_length = len(review_text)/maxLen

    scaled_length = len(review_text) / maxLen
    return [1, scaled_length]


# In[5]:


def Q1(dataset):
    # Get the max length of reviews
    maxLen = getMaxLen(dataset)

 
    # Create the feature matrix X and label vector y
    X = [featureQ1(d, maxLen) for d in dataset]
    y = [d['rating'] for d in dataset]

 
    theta, residuals, rank, s = numpy.linalg.lstsq(X, y, rcond=None)
    MSE = numpy.mean((X @ theta - y) ** 2)
    # Use residuals from lstsq for a more efficient MSE calculation
    MSE = residuals[0] / len(y)
    return theta, MSE


# In[6]:


### Question 2


# In[7]:


def featureQ2(datum, maxLen):
    # Implement (should be 1, length feature, day feature, month feature)
    feature = [1, len(datum['review_text'])/maxLen]

    feature = [1, len(datum['review_text']) / maxLen]
 
    # One-hot encoding for weekday (drop Monday)
    for i in range(1,7):
        feature.append(1) if datum['parsed_date'].weekday() == i else feature.append(0)

    weekday = datum['parsed_date'].weekday()
    for i in range(1, 7):
        feature.append(int(weekday == i))
 
    # One-hot encoding for month (drop January)
    month = datum['parsed_date'].month
    for i in range(2, 13):
        feature.append(1) if datum['parsed_date'].month == i else feature.append(0)

        feature.append(int(month == i))
 
    return feature

# In[8]:


def Q2(dataset):
    # Implement (note MSE should be a *number*, not e.g. an array of length 1)
    maxLen = getMaxLen(dataset)
    X2 = [featureQ2(d, maxLen) for d in dataset]
    Y2 = [d['rating'] for d in dataset]
    theta2 = numpy.linalg.lstsq(X2, Y2, rcond=None)[0]
    MSE2 = numpy.mean((X2 @ theta2 - Y2) ** 2)
    MSE2 = numpy.mean((numpy.array(X2) @ theta2 - Y2) ** 2)
    return X2, Y2, MSE2


# In[9]:


### Question 3


# In[10]:


def featureQ3(datum, maxLen):
    scaled_length = len(datum['review_text'])/maxLen
    scaled_length = len(datum['review_text']) / maxLen
    weekday = datum['parsed_date'].weekday()
    month = datum['parsed_date'].month
    return [1, scaled_length, weekday, month]

# In[11]:


def Q3(dataset):
    maxLen = getMaxLen(dataset)
    Y3 = [d['rating'] for d in dataset]

 
    X3 = [featureQ3(d, maxLen) for d in dataset]
    theta3, residuals, rank, s = numpy.linalg.lstsq(X3, Y3, rcond=None)
    MSE3 = numpy.mean((X3 @ theta3 - Y3) ** 2)

    MSE3 = residuals[0] / len(Y3)
 
    return X3, Y3, MSE3


# In[12]:


### Question 4


# In[13]:


def Q4(dataset):
    split_point = len(dataset) // 2
    train_set = dataset[:split_point]
    test_set = dataset[split_point:]

 
    maxLen = getMaxLen(train_set)
    Y_train = [d['rating'] for d in train_set]
    Y_test = [d['rating'] for d in test_set]

 
    # direct encoding
    X_train_direct = [featureQ3(d, maxLen) for d in train_set]
    theta_direct, residuals, rank, s = numpy.linalg.lstsq(X_train_direct, Y_train, rcond=None)

    theta_direct = numpy.linalg.lstsq(X_train_direct, Y_train, rcond=None)[0]
 
    X_test_direct = [featureQ3(d, maxLen) for d in test_set]
    predictions_direct = X_test_direct @ theta_direct
    predictions_direct = numpy.array(X_test_direct) @ theta_direct
    test_mse2 = numpy.mean((predictions_direct - Y_test) ** 2)

 
    # one-hot encoding
    X_train_onehot = [featureQ2(d, maxLen) for d in train_set]
    theta_onehot, residuals, rank, s = numpy.linalg.lstsq(X_train_onehot, Y_train, rcond=None)

    theta_onehot = numpy.linalg.lstsq(X_train_onehot, Y_train, rcond=None)[0]
 
    X_test_onehot = [featureQ2(d, maxLen) for d in test_set]
    predictions_onehot = X_test_onehot @ theta_onehot
    predictions_onehot = numpy.array(X_test_onehot) @ theta_onehot
    test_mse3 = numpy.mean((predictions_onehot - Y_test) ** 2)

 
    return test_mse2, test_mse3


# In[14]:


### Question 5


# In[15]:


def featureQ5(datum):
    review_text = datum.get('review/text', '')
    review_length = len(review_text)
    return [1, review_length]

# In[16]:


def Q5(dataset, feat_func):
    # Prepare data
    X = [feat_func(d) for d in dataset]
    y = [d['review/overall'] >= 4 for d in dataset]

    y = numpy.array([d['review/overall'] >= 4 for d in dataset])
 
    # Fit model
    model = linear_model.LogisticRegression(class_weight='balanced')
    model.fit(X, y)

 
    # Make predictions
    preds = model.predict(X)

 
    # Calculate TP, TN, FP, FN, BER
    TP = sum((p and l) for p, l in zip(preds, y))
    TN = sum((not p and not l) for p, l in zip(preds, y))
    FP = sum((p and not l) for p, l in zip(preds, y))
    FN = sum((not p and l) for p, l in zip(preds, y))

    TP = numpy.sum((preds == True) & (y == True))
    TN = numpy.sum((preds == False) & (y == False))
    FP = numpy.sum((preds == True) & (y == False))
    FN = numpy.sum((preds == False) & (y == True))
 
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0
    BER = 0.5 * (FPR + FNR)

 
    return TP, TN, FP, FN, BER


# In[17]:


### Question 6


# In[18]:


def Q6(dataset):
    # Prepare data
    X = [featureQ5(d) for d in dataset]
    y = [d['review/overall'] >= 4 for d in dataset]

 
    # Train the model
    model = linear_model.LogisticRegression(class_weight='balanced')
    model.fit(X, y)

 
    # Get predicted probabilities for the positive class
    probabilities = model.predict_proba(X)[:, 1]

 
    # Combine probabilities with true labels and sort by probabilities
    predictions_with_labels = list(zip(probabilities, y))
    predictions_with_labels.sort(key=lambda x: x[0], reverse=True)

 
    #  Calculate Precision@K for each K
    K = [1, 100, 1000, 10000]
    precs = []

 
    for k in K:
        # Get the top K items from the sorted list
        top_k = predictions_with_labels[:k]
        
         
        # Count how many of the top K are actually positive (True)
        # The label is the second element in our (score, label) pair
        true_positives_in_top_k = sum(1 for _, label in top_k if label)
        
         
        # Calculate precision@k
        precision_at_k = true_positives_in_top_k / k
        precs.append(precision_at_k)

 
    return precs


# In[19]:


### Question 7


# In[20]:


def featureQ7(datum):
    # Implement (any feature vector which improves performance over Q5)
    review_text = datum.get('review/text', '')
    review_length = len(review_text)
    review_taste = datum.get('review/taste', 0)
    beer_ABV = datum.get('beer/ABV', 0)
    return [1, review_length, review_taste, beer_ABV]
