# -*- coding: utf-8 -*-
"""
@author: Sumit Kumar
"""
from summarizer.k_means_cluster import cluster
from summarizer.bert_parent import BertParent
from summarizer.sentence_handler import SentenceHandler

def PreProcessor(body, summary_length,min_length : int = 40):
    
    model = BertParent('bert-large-uncased')
    algorithm='kmeans'
    sentence_handler =  SentenceHandler()
    random_state = 12345

    #We keep minimum and maximum word limit in a sentence to consider it for summary
    sentences = sentence_handler(body, min_length=40, max_length=600)
    print("Sentences filtered out based on min and max length")
    
    if sentences:
        #...hidden contains n*1024 matrix as word embeddings returned by BERT model where n is number of sentences
        hidden =model(sentences)
        #...we call k-means algorithm and pass the word embeddings done by the BERT model
        hidden_args = cluster(hidden, algorithm, random_state, summary_length)
        sentences = [sentences[j] for j in hidden_args]
        
    return ' '.join(sentences)
