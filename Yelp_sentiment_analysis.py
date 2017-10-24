################################################################################
# Created on Sep 22, 2017

#Author: Yashwanth
#Usage: python yelp_sentiment_analysis.py
################################################################################

# Import the pandas package, then use the "read_csv" function to read the labeled training data

import pandas as pd
df_clt_reviews = pd.read_csv("charlotte_restaurant_reviews.csv")
train=df_clt_reviews[df_clt_reviews['stars_x']==1]

#Understanding # rows and # cols
train.shape
#Column names
train.columns.values

train = train.reset_index()

#Sample one observation
example1= train['text'][0]
print example1

import re
from nltk.corpus import stopwords # Import the stop word list

def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = raw_review
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))   

# Get the number of reviews based on the dataframe column size
num_reviews = train["text"].size

# Initialize an empty list to hold the clean reviews
clean_train_reviews = []

for i in xrange( 0, num_reviews ):
    # If the index is evenly divisible by 10, print a message
    if((i+1)%1000== 0):
        print "Review %d of %d\n" % ( i+1, num_reviews)
        clean_train_reviews.append(review_to_words( train["text"][i]))

from sklearn.feature_extraction.text import CountVectorizer
# Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool
vectorizer = CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,stop_words = None,max_features = 50)

# fit_transform() does two functions: First, it fits the model and learns the vocabulary; 
# Second, it transforms our training data into feature vectors. The input to fit_transform should be a list of strings.
train_data_features = vectorizer.fit_transform(clean_train_reviews)
# Numpy arrays are easy to work with, so convert the result to an array
train_data_features = train_data_features.toarray()

import numpy as np

vocab = vectorizer.get_feature_names()
print vocab

# Sum up the counts of each vocabulary word
dist = np.sum(train_data_features, axis=0)

# For each, print the vocabulary word and the number of times it 
# appears in the training set
for tag, count in zip(vocab, dist):
    print count, tag


from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
sid=SIA()

sentiment_score=[]
for i in xrange( 0, num_reviews ):
    # If the index is evenly divisible by 10, print a message
    if((i+1)%1000== 0):
        print "Review %d of %d\n" % ( i+1, num_reviews)
    sentiment_score.append(sid.polarity_scores(train["text"][i]))    		

df_clt_reviews_sentiment = pd.read_csv("charlotte_restaurant_reviews_sentiment.csv")
train=df_clt_reviews_sentiment[df_clt_reviews_sentiment['compounded']>0]
