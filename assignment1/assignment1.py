import gzip
from collections import defaultdict
import string
from sklearn import linear_model

def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r

def readGz(path):
  for l in gzip.open(path, 'rt'):
    yield eval(l)

###### Task 1 ######
# Get parameters for training the model
def alphaUpdate(ratingsTrain, alpha, betaU, betaI, lamb):
    total_error = 0
    for user, item, rating in ratingsTrain:
        bu = betaU.get(user, 0)
        bi = betaI.get(item, 0)
        total_error += rating - (bu + bi)
    newAlpha = total_error / len(ratingsTrain)
    return newAlpha

def betaUUpdate(ratingsPerUser, alpha, betaU, betaI, lamb):
    newBetaU = {}
    for user in ratingsPerUser:
        total_error = 0
        Iu = ratingsPerUser[user]
        
        for (b, r) in Iu:
            bi = betaI.get(b, 0)
            total_error += (r - (alpha + bi))
        
        newBetaU[user] = total_error / (lamb + len(Iu))
    return newBetaU

def betaIUpdate(ratingsPerItem, alpha, betaU, betaI, lamb):
    newBetaI = {}
    for b in ratingsPerItem:
        total_error = 0
        Ui = ratingsPerItem[b]
        
        for (u, r) in Ui:
            bu = betaU.get(u, 0)
            total_error += (r - (alpha + bu))
        
        newBetaI[b] = total_error / (lamb + len(Ui))
    return newBetaI

# Train the model
def goodModel(ratingsTrain, ratingsPerUser, ratingsPerItem, alpha, betaU, betaI):
    lamb = 1.0
    N_iterations = 5

    print("Start model training.")
    for i in range(N_iterations):
        alpha = alphaUpdate(ratingsTrain, alpha, betaU, betaI, lamb)
        betaU = betaUUpdate(ratingsPerUser, alpha, betaU, betaI, lamb)
        betaI = betaIUpdate(ratingsPerItem, alpha, betaU, betaI, lamb)
    
    print("Training finished.")
    return alpha, betaU, betaI

# Write predictions to file
def writePredictionsRating(alpha, betaU, betaI):
    print("Write predictions to predictions_Rating.csv.")
    predictions = open("predictions_Rating.csv", 'w')
    for l in open("../datasets/pairs_Rating.csv"):
        if l.startswith("userID"):
            predictions.write(l)
            continue
        u,b = l.strip().split(',')
        
        bu = betaU.get(u, 0)
        bi = betaI.get(b, 0)

        prediction = alpha + bu + bi
        
        _ = predictions.write(u + ',' + b + ',' + str(prediction) + '\n')

    predictions.close()
    print("Done writing Rating predictions.")

###### Task 2 ######
def jaccardThresh(u, b, ratingsPerItem, ratingsPerUser):
    
    # Set of users who read book 'b'
    users_who_read_b = set([user for (user, rating) in ratingsPerItem.get(b, [])])
    
    # List of (book, rating) pairs read by user 'u'
    books_read_by_u = ratingsPerUser.get(u, [])
    
    maxSim = 0
    
    for (b_prime, r) in books_read_by_u:
        if b == b_prime: continue # Don't compare a book to itself
            
        # Set of users who read book 'b_prime'
        users_who_read_b_prime = set([user for (user, rating) in ratingsPerItem.get(b_prime, [])])
        
        # Calculate Jaccard similarity
        numer = len(users_who_read_b.intersection(users_who_read_b_prime))
        denom = len(users_who_read_b.union(users_who_read_b_prime))
        
        sim = 0
        if denom > 0:
            sim = numer / denom
            
        if sim > maxSim:
            maxSim = sim
    
    # Predict 1 if similarity is high OR if the book is popular
    if maxSim > 0.013 or len(ratingsPerItem.get(b, [])) > 40:
        return 1
    return 0

def writePredictionsRead(ratingsPerItem, ratingsPerUser):
    print("Writing predictions to predictions_Read.csv.")
    predictions = open("predictions_Read.csv", 'w')
    for l in open("../datasets/pairs_Read.csv"):
        if l.startswith("userID"):
            predictions.write(l)
            continue
        u,b = l.strip().split(',')
        
        # Call the Jaccard threshold function for each pair
        pred = jaccardThresh(u, b, ratingsPerItem, ratingsPerUser)
        
        _ = predictions.write(u + ',' + b + ',' + str(pred) + '\n')

    predictions.close()
    print("Done writing Read predictions.")

###### Task 3 ######
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

punctuation = set(string.punctuation)

# def featureCat(datum, wordId, wordSet):
#     feat = [0]*len(wordId)
#     r = ''.join([c for c in datum['review_text'].lower() if c not in punctuation])
#     for w in r.split():
#         if w in wordSet:
#             idx = wordId[w]
#             feat[idx] += 1
#     feat.append(1)
#     return feat

def writePredictionsCategory(pred_test):
    print("Writing predictions to predictions_Category.csv...")
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
    print("Done writing Category predictions.")


###### Main Execution ######
# Read and process data
allRatings = []
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
ratingsTrain = []

for user, book, rating in readCSV("../datasets/train_Interactions.csv.gz"):
    allRatings.append(rating)
    ratingsPerUser[user].append((book, rating))
    ratingsPerItem[book].append((user, rating))
    ratingsTrain.append((user, book, rating))

print("--- Starting Task 1: Rating Prediction ---")
globalAverage = sum(allRatings) / len(allRatings)
betaU = defaultdict(int)
betaI = defaultdict(int)
alpha = globalAverage

# Train the model
alpha, betaU, betaI = goodModel(ratingsTrain, ratingsPerUser, ratingsPerItem, alpha, betaU, betaI)

# Generate predictions
writePredictionsRating(alpha, betaU, betaI)
print("--- Task 1 Complete ---")

print("\n--- Starting Task 2: Read Prediction ---")
writePredictionsRead(ratingsPerItem, ratingsPerUser)
print("--- Task 2 Complete ---")

print("\n--- Starting Task 3: Category Prediction (Improved) ---")
print("Reading category training data...")
data_train = list(readGz("../datasets/train_Category.json.gz"))

reviews_train = [d['review_text'] for d in data_train]
y_train = [d['genreID'] for d in data_train]

vectorizer = TfidfVectorizer(stop_words='english', 
                           max_features=5000,
                           ngram_range=(1, 2))
print("Creating TF-IDF training matrix...")

X_train = vectorizer.fit_transform(reviews_train)

print("Training LinearSVC model...")
mod = LinearSVC(C=0.1, max_iter=2000, dual=True) 
mod.fit(X_train, y_train)

print("Reading category test data...")
data_test = list(readGz("../datasets/test_Category.json.gz"))
reviews_test = [d['review_text'] for d in data_test]

print("Creating TF-IDF test matrix...")

X_test = vectorizer.transform(reviews_test)

print("Making predictions...")
pred_test = mod.predict(X_test)

writePredictionsCategory(pred_test)
print("--- Task 3 Complete ---")