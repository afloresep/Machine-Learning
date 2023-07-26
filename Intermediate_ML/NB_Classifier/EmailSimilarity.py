from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

emails = fetch_20newsgroups()

print(emails.target_names)
#['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware'...

# Select categories we want our NB Classifier to difference
emails = fetch_20newsgroups(categories = ['rec.sport.baseball', 'rec.sport.hockey']) 

print(emails.data[5]) # See the content of the email
print(emails.target[5]) # returns 1, which corresponds to rec.sport.hockey

#Now we want to split our data into training and test sets. 
train_emails = fetch_20newsgroups(categories = ['rec.sport.baseball', 'rec.sport.hockey'], subset='train', shuffle = True, random_state = 108) 

test_emails = fetch_20newsgroups(categories = ['rec.sport.baseball', 'rec.sport.hockey'], subset='test', shuffle = True, random_state = 108) 

# Now we want to transform these emails into lists of word counts. the CountVectorizer class makes this easy for us. 

counter = CountVectorizer()

#  to tell counter what possible words can exist in our emails. counter has a .fit() a function that takes a list of all your data.
counter.fit(test_emails.data + train_emails.data)

# Now we make a list of the counts of our words in our training and test set. 
train_counts = counter.transform(train_emails.data)
test_counts = counter.transform(test_emails.data)


# Making a NB classifier that we can train and test on. training set and labels associated to the training emails. 
classifier = MultinomialNB()
classifier.fit(train_counts, train_emails.target)

print(classifier.score(train_counts, train_emails.target))


"""
Our classifier does a pretty good job distinguishing between soccer emails and hockey emails. But lets see how it does with emails about really different topics.

Find where you create train_emails and test_emails. Change the categories to be ['comp.sys.ibm.pc.hardware','rec.sport.hockey'].

Did your classifier do a better or worse job on these two datasets?
"""

