# -*- coding: utf-8 -*-
"""
@author: Cole Thompson
"""
#Import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt; plt.rcdefaults()
import re
from os import getcwd
from nltk.corpus import stopwords
from textblob import TextBlob, Word
from translate import Translator



#Configure pandas option to display all columns included within the dataframe
pd.set_option('display.max_columns',19)

class TweetDataHandler:
    """
    Class used for data extraction and pre-processing, imported in other modules for analysis and visualization
    of the tweet data.
    """

    def __init__(self):
        """
        Method for inializing configurations and data sources for the program. Additional data sources may be added
        to the dictionary, which contain the relevent path file type, and number of files for import. 
        
        The number of files for import is useful for testing purposes, as a means of reducing the time taken to run the program.
        The "IsImport" configuration will achieve a similar effect, where the program will not re-import data if already
        contained in cache. The "isMsg" configuration prints the status of program execution, also helpful while testing the program.

        The "isPreProc" configuration specifies whether the program will execute the pre-processing steps at run time, or retrieve the
        processed data from a file directory.

        TO DO: 1) Create timer decorator used for performance testing of different program features. The timer decorator funtion will include
        a condition that checks the configuration at runtime. 2) Convert dictionary to separate json file that is loaded
        to the program, so multiple json files can be created for easier variability of program configuration.
        """
        self.data_sources = {
            "troll_tweets" : [
                    ["\\data\\IRAhandle_tweets_",".csv",2],
                    ['external_author_id',
                     'author',
                     'content',
                     'region',
                     'language',
                     'publish_date',
                     'harvested_date',
                     'following',
                     'followers', 
                     'updates',
                     'post_type', 
                     'account_type', 
                     'new_june_2018', 
                     'retweet',
                     'account_category']]
            }
        
        self.cat_list = ['Fearmonger','Commercial', 'HashtagGamer','LeftTroll', 'NewsFeed', 'RightTroll']
        
        self.config = {
            "isMsg": True,
            "isTimed":False,
            "isPreProc":True,
            "isSavePreproc": False,
            "msg_import":'status : reading file into temp data frame',
            "msg_import_preproc":'status : reading preprocessed file into dataframe',
            "msg_append":'status : appending temp data to data frame',
            "msg_summary": 'status : summarizing data',
            "msg_calculations":'status : performing calculations',
            "msg_preproc_lower":'status : making tweets lower case',
            "msg_preproc_punct":'status : removing tweet punctuation',
            "msg_preproc_tags":'status : removing hashtags from tweets',
            "msg_preproc_stop":'status : removing stopwords from tweets',
            "msg_preproc_lemma":'status : performing tweet lemmatization',
            "msg_preproc_splitcols":'status : splitting dataframe column',
            "msg_preproc_save":'status : saving processed tweets',
            "msg_hash_links":'status : removing hyperlinks from tweets',
            "msg_preproc_sentiment":'status : calculating tweet sentiment',
            "msg_hash_tag":'status : retrieving and storing tweet hash tags for analysis',
            }

        #Get current directory for relative reference of data source paths
        self.my_path = getcwd()

    
    def msg_handle(self,message):
        """
        Method used to display program status messages. The method is called using the key for the
        dictionary with the configured status message. If the "isMsg" configuration is set to false,
        no message is printed at run-time.
        """

        #Check if isMsg is configured
        if self.config["isMsg"] == True:
            #Condition to check if message is in dictionary
            if message in self.config:

                #Print the message for the provided key value
                print(self.config[message])
                
    def get_csv_data(self,data_source):
        """
        Function to retrieve CSV data source, based on information contained within the config dictionary.
        """

        tweet_object = self.data_sources[data_source]
        #File path extension index in data_sources values
        file_path = tweet_object[0][0]
       
        #File extension index in data_sources values
        file_ext = tweet_object[0][1]
        
        #Number of files index in data_sources values
        number_of_files = tweet_object[0][2]
        data = pd.DataFrame()

        #Loop through file path for number of files to be imported
        file_num = 1
        
        while file_num <= number_of_files:
                self.msg_handle("msg_import")

                #Read csv file into dataframe
                df = pd.read_csv(self.my_path + file_path + str(file_num) + file_ext)
                
                #Append csv file to dataframe
                data = data.append(df)
                self.msg_handle("msg_append")
                file_num += 1
        return data

    def get_hash_tags(self,df,df_col):
        """
        Function to retrieve hash tags from data. Returns a series of hashtags to join to the troll tweet datafrme
        by index. Also returns a set of distinct hash tags.
        """
        self.msg_handle("msg_hash_tag")
        hash_tags_list = []
        
        for i in df[df_col]:
            try:
                #Regular expression to locate all hash tag values
                hash_tag = re.findall(r"#(\w+)",i)
                
                #Append hash_tags list with a list of each tweet's hash tags
                hash_tags_list.append(hash_tag)
            except TypeError:
                continue

        #Create pandas series to join to troll tweet dataframe   
        hash_tag_series = pd.Series(hash_tags_list, name='hash_tags')

        #List comprehension to extract hash tags from nested lists, and a set is constructed to filter for distinct values
        distinct_hash_tags = set([item for sublist in hash_tags_list for item in sublist])

        return (hash_tag_series, distinct_hash_tags)
    
    def get_cat_hash_tags(self):
        """
        Function to return hashtags by account category
        """
        hashtag_set_list = []
        for i in self.cat_list:
            cat_hashtags = self.troll_tweet_df[self.troll_tweet_df['account_category'] == i]['hash_tags'].tolist()
            hashtag_set_list.append(set([item for sublist in cat_hashtags for item in sublist]))
        
        return hashtag_set_list
            

    def put_hash_tags(self,df,df_col):
        """
        Method to join hash_tag_series to troll tweet dataframe by index, so that dataframe contains
        the hashtags contained within each tweet.
        """
        #Call get_hash_tags function to return hash_tag_series and distinct hashtags
        data = self.get_hash_tags(df,df_col)

        #Upack hash_tag_series from tuple returned by get_hash_tags function
        hash_tag_series = data[0]

        #Upack distinct_hash_tags from tuple returned by get_hash_tags function
        distinct_hash_tags = data[1]

        #Join hash_tag_series to dataframe
        df = pd.concat([df, hash_tag_series], axis=1, join='inner')
        distinct_hash_tags = pd.Series(list(distinct_hash_tags),name='distinct_hashtags')
        
        return (df,distinct_hash_tags)
    
    def run_troll_tweets(self):
        """
        Method used to get csv data for the configured "troll_tweets" data source, update
        dataframe with hash tags, and extract list of distinct hashtags contained within the
        tweet content.
        """
        #Get CSV data for troll tweets
        self.troll_tweet_data = self.get_csv_data('troll_tweets')
        
        #Update dataframe with hashtags for each tweet
        self.troll_tweets = self.put_hash_tags(self.troll_tweet_data,"content")
        
        #Assign troll_tweet_df to dataframe returned from put_hash_tags function
        self.troll_tweet_df = self.troll_tweets[0]

        #Assign distinct_hash_tags to list of distinct hash tags returned from put_hash_tags function
        self.distinct_hash_tags = self.troll_tweets[1]
        
        #Store distinct hash tags for later use
        if self.config["isSavePreproc"] == True:
            self.distinct_hash_tags.to_csv(self.my_path + "\\data\\" + "Processed\\distinct_hashtags.csv")
        
        return (self.troll_tweet_df, self.distinct_hash_tags)
    
    def get_tweet_sentiment(self, tweet):
        """
        Function to retrieve tweet sentiment for tweets using TextBlob
        """
        analysis = TextBlob(tweet)
        
        return list(analysis.sentiment_assessments)

        
    def tweet_pre_processing(self):
        """
        Method used for manipulating tweet content for NLP methods and text analytics.
        Includes steps for:
        1) Removing hyperlinks
        2) Removing hash tags
        3) Remove punctuation
        5) Make all words lower case
        6) Remove Stop-words
        7) Lemmatization
        8) Sentiment Analysis
        """
   
        
        #Condition to check if pre-processing is enabled, if false method retrieves processed data from csv file location
        if self.config["isPreProc"] == True:
            
            #Run troll tweet process
            self.tweedle_collection = self.run_troll_tweets()
            self.troll_tweet_df = self.tweedle_collection[0]
            self.distinct_hashtags = self.tweedle_collection[1]
            
            #Copy twitter data for pre-processing
            self.troll_tweet_df = self.troll_tweet_df[self.troll_tweet_df['language'] == 'English']

            #Filter null tweet content
            self.processed_tweets = pd.Series(self.troll_tweet_df['content'],index=self.troll_tweet_df.index, name='processed_content')
            
            #Removing hyperlinks from twitter data, regular expression used to identify URLS, placed before hash tag and punctuation removal to effectively do so
            self.msg_handle("msg_preproc_links")
            self.processed_tweets = self.processed_tweets.str.replace(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*','')
            
            #Removing hash tags from twitter data, regular expression used to replace tweets with nothing
            self.msg_handle("msg_preproc_tags")
            self.processed_tweets = self.processed_tweets.str.replace('#(\w+)','')
            
            #Removing stopwords from twitter data
            self.msg_handle("msg_preproc_stop")
            stop = stopwords.words('english')
            self.processed_tweets = self.processed_tweets.apply(lambda x: " ".join(x for x in x.split() if x not in stop))
            
            #Removing punctuation from twitter data, regular expression used to replace punctuation with nothing
            self.msg_handle("msg_preproc_punct")
            self.processed_tweets = self.processed_tweets.str.replace('[^\w\s]','')
                       
            #Removing null tweets, after applying preprocessing for other pre-processing methods
            self.processed_tweets = self.processed_tweets[self.processed_tweets.isna() == False]
            
            #Making twitter data lower case
            self.msg_handle("msg_preproc_lower")
            self.processed_tweets= self.processed_tweets.apply(lambda x: " ".join(x.lower() for x in x.split()))
        
            
            #Lemmatization
            self.msg_handle("msg_preproc_lemma")
            self.processed_tweets = self.processed_tweets.apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
                       
            
            #Join processed tweets to dataframe
            self.troll_tweet_df = self.troll_tweet_df.join(self.processed_tweets, how='left')
            
            #Get Tweet Sentiment - polarity, subjectivity and textblob assessments
            self.msg_handle("msg_preproc_sentiment")
            self.troll_tweet_df['sentiment'] = self.troll_tweet_df['processed_content'].apply(self.get_tweet_sentiment)
            
            #Split tweet sentiment containing list  into muliple columns
            self.msg_handle("msg_preproc_splitcols")
            self.troll_tweet_df[['sent_polarity', 'sent_subjectivity', 'sent_assessments']] = pd.DataFrame(self.troll_tweet_df.sentiment.values.tolist(), index=self.troll_tweet_df.index)

            
            #Store processed tweets to CSV for later use
            if self.config["isSavePreproc"] == True:
                self.msg_handle("msg_preproc_save")
                self.troll_tweet_df.to_csv(self.my_path + "\\data\\" + "Processed\\processed_tweets.csv")
            
        else:
            self.msg_handle("msg_import_preproc")
            self.troll_tweet_df = pd.read_csv(self.my_path + "\\data\\" + "Processed\\processed_tweets.csv")
            self.distinct_hashtags = pd.read_csv(self.my_path + "\\data\\" + "Processed\\distinct_hashtags.csv")
            
        return (self.troll_tweet_df, self.distinct_hashtags)
       