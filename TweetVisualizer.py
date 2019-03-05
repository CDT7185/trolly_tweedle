# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 10:01:00 2019

@author: Cole Thompson
"""
#Import modules
import TweetDataHandler as tdh
import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from collections import Counter
import matplotlib.pyplot as plt


class TweetVisualizer:
    """
    Class containing methods and attributes for generating visuals of the tweet content retrieved by the TweetDataHandler class.
    """
    def __init__(self):
        """
        Initializes the tweet visualizer class, setting the tweedle_collection attribute to data retrieved from the TweetDataHandler class. From the
        tweedle_collection attribute, the data frame from the troll tweets is derived, as well as the distinct hash tags contained within the tweet
        content. The tweet_pre_processing method is also executed, which performs pre-processing steps on the tweets for later NLP methods. 
        """
        self.tweedle = tdh.TweetDataHandler()
        self.tweedle_collection = self.tweedle.run_troll_tweets()
        self.troll_tweet_df = self.tweedle_collection[0]
        self.distinct_hashtags = self.tweedle_collection[1]
        self.processed_tweets = self.tweedle.tweet_pre_processing()
   
    def hbar_tweets_by_col(self,column):
        """
        Method to generate a horizontal bar chart, displaying a numerical value (number of tweets) grouped by the parameter column
        passed to the function.
        """
        #Create acct category data frame for matplotlib figure 
        df = pd.DataFrame(self.troll_tweet_df[column].value_counts())
        df = df.reset_index()
        
        #Setup & show matplot lib figure for number of tweets per column
        y_pos = np.arange(len(df['index']))
        number_of_tweets = df[column]
        plt.barh(y_pos, number_of_tweets, align='center')
        plt.yticks(y_pos,df['index'])
        plt.xlabel('Number of Tweets')

        #Display title containing the column used to group the numerical "number of tweets" data
        plt.title('Russian Troll Tweets by ' + str(column))
        plt.show()
    
    def wordcloud_hash_tags(self, account_category=None):
        """
        Method to generate a wordcloud containing the most frequently occurring hash tags in the tweets. There is an optional 
        account_category parameter that allows for the word cloud to filter the dataframe by the specified troll account
        category.
        """
        #Condition used to filter the dataframe according to the optional "account_category" parameter.
        #List comprehension  is used to create a list of each hash tag sublist contained in the troll tweet data frame
        if account_category == None:
            tags = [tag.lower() for row in self.troll_tweet_df['hash_tags'] for tag in row]
        else:
            tags = [tag.lower() for row in self.troll_tweet_df[self.troll_tweet_df['account_category'] == account_category]['hash_tags'] for tag in row]
       
        #Create a dictionary containing a count for each hash tag contained within the "hash tags" list
        word_cloud_dict = Counter(tags)
        wordcloud = WordCloud(max_font_size=50, max_words = 100, background_color='white').generate_from_frequencies(word_cloud_dict)
        plt.figure()
        plt.imshow(wordcloud, interpolation='bilinear')

        #Turn off axis as it there is none used for a wordcloud visualization
        plt.axis("off")
        plt.title("Hash Tag Word Cloud : Category - " + str(account_category))
        plt.show()
    
    def wordcloud_tweets(self, account_category=None):
        """
        Method to generate a wordcloud containing the most frequent words in the tweet content. There is an optional 
        account_category parameter that allows for the word cloud to filter the dataframe by the specified troll account
        category.
        """
        #Condition used to filter the dataframe according to the optional "account_category" parameter.
        #Filter out null values, which cause an issue for the wordcloud constructor, some tweets do not have content following
        #the pre-processing steps. i.e Content solely includes URL and hashtags, etc.

        if account_category is not None:
            tweet_content = self.processed_tweets[self.processed_tweets['account_category'] == account_category]
            tweet_content = tweet_content[tweet_content['content'].isna() == False]['content']
            
        else:
            tweet_content = self.processed_tweets[self.processed_tweets['content'].isna() == False]['content']
        wordcloud = WordCloud(max_font_size=50, max_words = 100, background_color='white').generate(' '.join(tweet_content))
        plt.figure()
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title("Tweet Content Cloud : Category -  " + str(account_category) )
        plt.show() 
