import pandas as pd
import numpy as np


class TweetCalculator:
    def __init__(self):
        """
        Initialize data for calculations performed on tweets, a dataframe is passed to the constructor.
        """

    def calc_followers_and_following(self, df):
        """
        Calculates and prints followers/following information for troll twitter handles.
        """
        self.author_df = df[['author', 'publish_date', 'account_category']]
        self.author_follow_df = df[['author', 'publish_date', 'account_category', 'following', 'followers']]

        # Create tweet author df, grouping by the max tweet date
        self.author_df = self.author_df.groupby(by=['author', 'account_category'], as_index=False).max()

        # Join followers info by author, account_category and publish date, getting latest followers for each tweeter
        self.tweeters_and_followers = self.author_df.merge(self.author_follow_df,
                                                           on=['author', 'account_category', 'publish_date'],
                                                           how='left')

        # Drop duplicates from data frame
        self.tweeters_and_followers = self.tweeters_and_followers.drop_duplicates(inplace=False)

        # Create arrays of followers/following
        self.followers_array = np.array(self.tweeters_and_followers['followers'].values)
        self.following_array = np.array(self.tweeters_and_followers['following'].values)

        # Descriptive statistics for  followers/following of troll accounts at max date of tweet per troll acct
        print("Russian Troll Tweets : Followers/Following")
        self.followers_avg = np.mean(self.followers_array)
        print("Followers Avg : " + str(int(self.followers_avg)))
        self.followers_median = np.median(self.followers_array)
        print("Followers Median : " + str(int(self.followers_median)))
        self.following_avg = np.mean(self.following_array)
        print("Following Avg : " + str(int(self.following_avg)))
        self.following_median = np.median(self.following_array)
        print("Following Median : " + str(int(self.following_median)))

        # Compute followers sum and following sum for followers-to-following ratio
        self.followers_sum = np.sum(self.followers_array)
        self.following_sum = np.sum(self.following_array)
        self.follow_ratio = round(np.divide(self.followers_sum, self.following_sum), 2)
        print("Followers-to-following : " + str(self.follow_ratio))