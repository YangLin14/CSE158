# Assignment 1:

## Task 1: Book Rating Prediction

-   **Goal:** To make predictions about how a user would rate a book in terms of star rating using the historical data.

-   **Strategy:** I use the _latent factor model_ to make **prediction**:

    $$
    prediction = \alpha + b_u + b_i
    $$

    Which components represent:

    -   $\alpha$: The **global average** of the book ratings.
    -   $b_u$: The **user bias** on predicting books (how much the user's ratings typically deviate from the global average).
    -   $b_i$: The **item bias** on being predicted by different users (how much the item's ratings typically deviate from the global average).

-   **Steps for completing Task 1:**
    1. Read the training data `train_Interactions.csv.gz`.
    2. Train the model with this data using the strategy.
        - `goodModel` function runs the coordinate descent for 5 iterations to optimize $\alpha$, $\beta_u$, and $\beta_i$.
    3. Generate the predictions and write it to `predictions_Rating.csv`.

## Task 2: Read Prediction

-   **Goal:** Predict whether a user would read the specific book (0 or 1).

-   **Strategy:** I use the Jaccard similarity appoarch.

    -   The Jaccard similarity is a simple formula that measures the overlap between these two sets:

        $$
        \text{Similarity} = \frac{\text{Size of Intersection}}{\text{Size of Union}}
        $$

        -   **Intersection:** The number of users who read both "Book A" and "Book B". This is the "overlap" in your audience.
        -   **Union:** The number of users who read at least one of the books ("Book A" or "Book B" or both). This is the "total" audience.

    -   This method checks if the target book (b) is similar to the other books the user (u) has read.
    -   Similarity score is measured under the idea of "similar books are read by similar people."
        -   For example, if _any_ book that the user (u) has already read is highly similar to the target book (b), the model concludes that "this user likes books that are read by same people who read target book (b) -> predict 1.
    -   It predicts 1 if the max similarity is above a threshold (0.013) or if the book is very popular (read more than 40 times).

-   **Steps for completing Task 2:**
    1. Implement `jaccardThresh` function to get the similarity of the target book and each of those books in the user's history.
    2. Generate the predictions and write it to `predictions_Read.csv`.

## Task 3: Category prediction

-   **Goal:** Predict the genre of the book based on the text of a review.

-   **Strategy:** Use the "Bag of Words" appoarch.

    -   Scan through all review texts in `train_Category.json.gz` to get the 2,000 most frequent words.
    -   Turn each review into vector to count how many times each of the top words appear in that review.
    -   Then use logistic regression model to train on the data and learn which words correspond to which genres.
    -   Finally, process the `test_Category.json.gz` reviews using the same 2,000 word list, turn them into vectors, and feed them to the trained model to get the predictions.

-   **Steps for completing Task 3:**
    1. Find the top 2,000 words.
    2. Create Training Matrix X_train and Labels y_train.
    3. Train the Classifier.
    4. Create Test Matrix X_test.
    5. Make Predictions and write it to `predictions_Category.csv`.
