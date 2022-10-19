# -*- coding: utf-8 -*-

"""# Clean text"""

import re
import nltk
import pandas as pd
import numpy as np
import os
import string
from multiprocessing import Process, freeze_support
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

import warnings
warnings.filterwarnings('ignore')

pd.options.display.max_colwidth = 50

# The path to the output folder where all the outputs will be saved
output_path = "/Users/martinm1/Documents/Social Media Analysis/McPlant/Documents"

def clean(text):
    """ Removes from the input text:
        - html tags, 
        - punctuations, 
        - stop words,
        - words of less than 3 characters
        - all words that are not a noun, an adjective, or a verb.

    Arguments:
        text (str) :
            The text to be cleaned

    Raises:
        TypeError: if input parameters are not of the expected type

    Returns:
        text (str) :
            The cleaned text (lowercased)
    """
    
    if not isinstance(text, str):
        raise TypeError("Argument 'text' must be a string.")
    
    # Strip the text, lowercase it, and remove the HTML tags and punctuations
    text = text.lower().strip()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^ 0-9a-z]', ' ', text)
    text = re.sub(r'\b(\d+\d)\b', '', text)
    text = re.sub(r'http|https|www', '', text)
    text = re.sub(r'\b[a-z]\b', '', text)
    text = re.sub(r' +', ' ', text)
    text = text.translate(text.maketrans('', '', string.punctuation)) #extra punctuations removal

    # Remove all the stop words
    stop_words = nltk.corpus.stopwords.words('english')
    stop_words.extend([
        'from', 're', 'also'
    ])
    stop_words = {key: True for key in set(stop_words)}
    
    # Keep only specific pos (part of speech: nouns, adjectives, and adverbs)
    keep_pos = {key: True for key in ['NN','NNS','NNP','NNPS', 'JJ', 'JJR', 'JJS','RB','RBR','RBS']}
    
    return " ".join([word 
                    for word, pos in nltk.tag.pos_tag(text.split()) 
                    if len(word) > 2 and word not in stop_words and pos in keep_pos])

def lemmatize(text: str, lemmatizer: nltk.stem.WordNetLemmatizer) -> str:
    """ Lemmatize the words in a sentence by:
        - mapping the POS tag to each word,
        - lemmatize the word.

    Arguments:
        sentence (str):
            The sentence in which the words need to be lemmatized
        lemmatizer:
            Lemmatizer function

Raises:
        TypeError: if input parameters are not of the expected type

    Returns:
        Lemmatized text
    ----------------------------------------------------------------------------------------
    """
    
    if not isinstance(text, str):
        raise TypeError("Argument 'text' must be a string.")

    lemmas = []
    tag_dict = {
        "J": nltk.corpus.wordnet.ADJ,
        "N": nltk.corpus.wordnet.NOUN,
        "V": nltk.corpus.wordnet.VERB,
        "R": nltk.corpus.wordnet.ADV
    }
    
    tokenized_words = nltk.word_tokenize(text)
    for tokenized_word in tokenized_words:
        tag = nltk.tag.pos_tag([tokenized_word])[0][1][0].upper() # Map POS tag to first character lemmatize() accepts
        wordnet_pos = tag_dict.get(tag, nltk.corpus.wordnet.NOUN)
        lemma = lemmatizer.lemmatize(tokenized_word, wordnet_pos)
        lemmas.append(lemma)
    
    return " ".join(lemmas)

articles = pd.read_csv('/Users/martinm1/Documents/Social Media Analysis/McPlant/Documents/McPlant Nov 2020 to Oct 2022_timeseries.csv')

print(f"Number of tweets: {len(articles)}")

articles.head()

articles.dtypes

"""Number of words per tweet"""

articles["n_words"] = articles["tweet"].apply(lambda text: len(text.split(" ")))

import matplotlib.pyplot as plt

# clean article word count
#plt.figure(figsize=(20, 10))
#plt.hist(articles[articles["n_words"] < 150].n_words, bins = 200, color = ["#bf7cbb"])
#plt.gca().set(xlim=(-10, 70), ylabel='Number of tweets', xlabel='Number of words')
#plt.box(False)
#plt.title('Number of words per tweet', fontdict=dict(size=24))
#plt.show()



# Clean the tweets
articles["article_clean"] = articles["tweet"].apply(clean)

lemmatizer = nltk.stem.WordNetLemmatizer()

articles["article_clean"] = articles["article_clean"].apply(lambda x: lemmatize(x, lemmatizer)) 

articles["n_words_clean"] = articles["article_clean"].apply(lambda x: len(x.split(" ")))

articles.head()

# number words per clean article
#import matplotlib.pyplot as plt

# clean article word count
#plt.figure(figsize=(20, 10))
#plt.hist(articles[articles["n_words_clean"] < 400].n_words_clean, 
    #  bins = 200, 
    #   color = ["#bf7cbb"])
#plt.xlim((-10, 80))
#plt.xlabel("Article word count", fontsize=18)
#plt.ylabel("Number of articles", fontsize=18)
#plt.box(False)
#plt.show()

articles.to_csv(f"{output_path}/mcplant_article_clean.csv", index=False)

"""# Topic Modeling Packages"""


import gensim
from gensim import corpora
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt

pd.options.display.max_colwidth = 200
import warnings
warnings.filterwarnings('ignore')

"""## Train LDA model"""

words = [text.split() for text in articles['article_clean']]

# create the term dictionary of courpus
dictionary = corpora.Dictionary(words)

# filter the least and most frequent words: filters if less than no_below, more than no_above
#dictionary.filter_extremes(no_below=10, no_above=0.9) 
#dictionary.compactify()

# convert list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(word) for word in words]

"""## Coherence Score Calculations"""

# train LDA, computing the coherence score for a range of topics
coherence_scores = []

for num_topics in range(2, 30, 1):
    
    print(f"Number of topics: ", num_topics)
    
    # create the object for LDA model using gensim library

    Lda = gensim.models.ldamulticore.LdaMulticore

    # run and train LDA model on the document term matrix.
    ldamodel = Lda(doc_term_matrix, 
                num_topics=num_topics, 
                id2word = dictionary, 
                passes=20, 
                chunksize = 100, 
                random_state=42,
                workers=6)
    
    # compute the coherence score
   
    coherence_model = CoherenceModel(model=ldamodel, 
                                        texts=words, 
                                        dictionary=dictionary, 
                                        coherence='c_v')

    coherence_lda = coherence_model.get_coherence()
    
    coherence_scores.append((num_topics, coherence_lda))

    coherence_scores = [*zip(*coherence_scores)]

# plot the coherence score for topics 2 step - 30
plt.plot(coherence_scores[0], coherence_scores[1], marker='o')
plt.title('Coherence Score for Topics')
plt.show()

"""# Run LDA Model"""

# set the number of topics where coherence score is the highest
num_topics = 25
# run and train LDA model on the document term matrix.
Lda = gensim.models.ldamulticore.LdaMulticore

ldamodel = Lda(doc_term_matrix, 
            num_topics=num_topics, 
            id2word=dictionary, 
            passes=40, 
            chunksize=100, 
            random_state=100,
            workers=6)

# view the topics with their most important words and their proportions
ldamodel.print_topics(num_topics=num_topics, num_words=10)

"""## Visualization"""
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
pyLDAvis.enable_notebook()
# visualize the intractive LDA plot 30
lda_display = pyLDAvis.gensim_models.prepare(ldamodel, 
                                    doc_term_matrix, 
                                    dictionary, 
                                    sort_topics=False)
pyLDAvis.display(lda_display)

# save the plot in html format
pyLDAvis.save_html(lda_display, f"{output_path}/mcplant_topic_{num_topics}.html")

lda_display.topic_info

lda_display.topic_info.to_excel('pyldavis_wordfreq_gen_from mcplant 25.xlsx')

"""## Dominant Topics"""

# user inputs
corpus = doc_term_matrix
texts = articles
df = articles

# function to get dominant topic, percentage of contribution, and keywords for each document
def format_topics_sentences(ldamodel, corpus):

    results = []
    
    # get main topic in each document
    for row in ldamodel[corpus]:
        
        if len(row) == 0:
            continue
            
        row = list(sorted(row, key=lambda elem: elem[1], reverse=True))
        
        # get the dominant topic, percentage of contribution and keywords for each document
        topic_num, prop_topic = row[0]        
        wp = ldamodel.show_topic(topic_num)
        topic_keywords = ", ".join([word for word, prop in wp])
        results.append((topic_num, round(prop_topic, 4), [topic_keywords]))
    
    df = pd.DataFrame.from_records(results, columns=['dominant_topic', 'weight', 'keywords'])
    
    return(df)

df_topics = format_topics_sentences(ldamodel, corpus)
df_topics.head()

# concatenate with the main dataset
articles = pd.concat([articles, df_topics.reindex(articles.index)], axis=1)

articles.head()