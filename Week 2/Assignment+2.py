
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.0** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# # Assignment 2 - Introduction to NLTK
# 
# In part 1 of this assignment you will use nltk to explore the Herman Melville novel Moby Dick. Then in part 2 you will create a spelling recommender function that uses nltk to find words similar to the misspelling. 

# ## Part 1 - Analyzing Moby Dick

# In[1]:

import nltk
import pandas as pd
import numpy as np

# If you would like to work with the raw text you can use 'moby_raw'
with open('moby.txt', 'r') as f:
    moby_raw = f.read()
    
# If you would like to work with the novel in nltk.Text format you can use 'text1'
moby_tokens = nltk.word_tokenize(moby_raw)
text1 = nltk.Text(moby_tokens)


# ### Example 1
# 
# How many tokens (words and punctuation symbols) are in text1?
# 
# *This function should return an integer.*

# In[2]:

def example_one():
    
    return len(nltk.word_tokenize(moby_raw)) # or alternatively len(text1)

example_one()


# ### Example 2
# 
# How many unique tokens (unique words and punctuation) does text1 have?
# 
# *This function should return an integer.*

# In[3]:

def example_two():
    
    return len(set(nltk.word_tokenize(moby_raw))) # or alternatively len(set(text1))

example_two()


# ### Example 3
# 
# After lemmatizing the verbs, how many unique tokens does text1 have?
# 
# *This function should return an integer.*

# In[4]:

from nltk.stem import WordNetLemmatizer

def example_three():

    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w,'v') for w in text1]

    return len(set(lemmatized))

example_three()


# ### Question 1
# 
# What is the lexical diversity of the given text input? (i.e. ratio of unique tokens to the total number of tokens)
# 
# *This function should return a float.*

# In[11]:

def answer_one():
    unique_tokens = example_three()
    total_tokens = len(text1)
    lexical_diversity = unique_tokens/total_tokens
    
    return lexical_diversity

answer_one()


# ### Question 2
# 
# What percentage of tokens is 'whale'or 'Whale'?
# 
# *This function should return a float.*

# In[14]:

from nltk import FreqDist
def answer_two():
    distribution = FreqDist(text1)
    Whale_freq = distribution[u'Whale']
    whale_freq = distribution[u'whale']
    total_tokens = len(text1)
    freq = (Whale_freq + whale_freq)/total_tokens
    freq_percentage = freq*100
    return freq_percentage

answer_two()


#another_code:    
    #test = [w for w in text1 if re.search(r'^[Ww]hale$',w)] 
    #return ( len(test)/len(text1) ) * 100


# ### Question 3
# 
# What are the 20 most frequently occurring (unique) tokens in the text? What is their frequency?
# 
# *This function should return a list of 20 tuples where each tuple is of the form `(token, frequency)`. The list should be sorted in descending order of frequency.*

# In[16]:

def answer_three():
    
    distribution = FreqDist(text1)
    return distribution.most_common(20) 

answer_three()


# ### Question 4
# 
# What tokens have a length of greater than 5 and frequency of more than 150?
# 
# *This function should return a sorted list of the tokens that match the above constraints. To sort your list, use `sorted()`*

# In[17]:

def answer_four():
    
    distribution = FreqDist(text1)
    
    return sorted(w for w in set(text1) if len(w) > 5 and distribution[w] > 150)

answer_four()


# ### Question 5
# 
# Find the longest word in text1 and that word's length.
# 
# *This function should return a tuple `(longest_word, length)`.*

# In[18]:

def answer_five():
    
    text_list = list(text1)
    sorted_list = sorted(text_list, key=len)
    longest_word = sorted_list[-1]
    return (longest_word, len(longest_word))

answer_five()


# ### Question 6
# 
# What unique words have a frequency of more than 2000? What is their frequency?
# 
# "Hint:  you may want to use `isalpha()` to check if the token is a word and not punctuation."
# 
# *This function should return a list of tuples of the form `(frequency, word)` sorted in descending order of frequency.*

# In[19]:

def answer_six():
    
    distribution = FreqDist(text1)
    words = distribution.keys()
    freqwords = [(distribution[word], word) for word in words if word.isalpha() and distribution[word] > 2000]
    sorted_words = sorted(freqwords, key=lambda x:x[0], reverse=True)
    
    return sorted_words

answer_six()


# ### Question 7
# 
# What is the average number of tokens per sentence?
# 
# *This function should return a float.*

# In[24]:

def answer_seven():
    
    sentences = nltk.sent_tokenize(moby_raw)
    tokens_per_sentence = [len(nltk.word_tokenize(sentence)) for sentence in sentences]
    #print(tokens_per_sentence)
    number_of_sentences = len(tokens_per_sentence)
    total_tokens = sum(tokens_per_sentence)
    return total_tokens/number_of_sentences

    
answer_seven()


# ### Question 8
# 
# What are the 5 most frequent parts of speech in this text? What is their frequency?
# 
# *This function should return a list of tuples of the form `(part_of_speech, frequency)` sorted in descending order of frequency.*

# In[28]:

def answer_eight():
    parts_of_speech = nltk.pos_tag(text1)
    count = nltk.FreqDist(tag for (word, tag) in parts_of_speech)
    answer = count.most_common()[:6]
    output = [i for i in answer if i[0]!=',']
    return output
answer_eight()


# ## Part 2 - Spelling Recommender
# 
# For this part of the assignment you will create three different spelling recommenders, that each take a list of misspelled words and recommends a correctly spelled word for every word in the list.
# 
# For every misspelled word, the recommender should find find the word in `correct_spellings` that has the shortest distance*, and starts with the same letter as the misspelled word, and return that word as a recommendation.
# 
# *Each of the three different recommenders will use a different distance measure (outlined below).
# 
# Each of the recommenders should provide recommendations for the three default words provided: `['cormulent', 'incendenece', 'validrate']`.

# In[29]:

from nltk.corpus import words

correct_spellings = words.words()


# ### Question 9
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) on the trigrams of the two words.**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[30]:

def answer_nine(entries=['cormulent', 'incendenece', 'validrate']):
    recommendations= []
    dist_list = []
    for entry in entries:
        recommendation = ""
        shortest_dist = float('inf')
        for correct_spelling in correct_spellings:
            if list(entry.lower())[0] == list(correct_spelling.lower())[0]:
                entry_ngram = nltk.ngrams(entry, n=3)
                correct_ngram = nltk.ngrams(correct_spelling, n=3)
                jac_dist = nltk.jaccard_distance(set(entry_ngram), set(correct_ngram))
                if jac_dist < shortest_dist:
                    shortest_dist = jac_dist
                    recommendation = correct_spelling
        
        recommendations.append(recommendation)
        
    
    return recommendations
    
answer_nine()


# ### Question 10
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) on the 4-grams of the two words.**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[31]:

def answer_ten(entries=['cormulent', 'incendenece', 'validrate']):
    
    recommendations= []
    dist_list = []
    for entry in entries:
        recommendation = ""
        shortest_dist = float('inf')
        for correct_spelling in correct_spellings:
            if list(entry.lower())[0] == list(correct_spelling.lower())[0]:
                entry_ngram = nltk.ngrams(entry, n=4)
                correct_ngram = nltk.ngrams(correct_spelling, n=4)
                jac_dist = nltk.jaccard_distance(set(entry_ngram), set(correct_ngram))
                if jac_dist < shortest_dist:
                    shortest_dist = jac_dist
                    recommendation = correct_spelling
        
        recommendations.append(recommendation)
    return recommendations
    
answer_ten()


# ### Question 11
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Edit distance on the two words with transpositions.](https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance)**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[32]:

def answer_eleven(entries=['cormulent', 'incendenece', 'validrate']):
    recommendations= []
    dist_list = []
    for entry in entries:
        recommendation = ""
        shortest_dist = float('inf')
        for correct_spelling in correct_spellings:
            if list(entry.lower())[0] == list(correct_spelling.lower())[0]:
                #entry_ngram = nltk.ngrams(entry, n=4)
                #correct_ngram = nltk.ngrams(correct_spelling, n=4)
                edit_dist = nltk.edit_distance(entry, correct_spelling)
                if edit_dist < shortest_dist:
                    shortest_dist = edit_dist
                    recommendation = correct_spelling
        
        recommendations.append(recommendation)
    return recommendations 
answer_eleven()


# In[ ]:



