import TweetVisualizer as visual
import TweetCalculator as calculator
import numpy as np

def main():
    #Create list for looping, load data and initialize tweet visualizer/calculator
    categories = [None,'Fearmonger','Commercial', 'HashtagGamer','LeftTroll', 'NewsFeed', 'RightTroll']
    viz = visual.TweetVisualizer()
    calc = calculator.TweetCalculator(viz.troll_tweet_df)
    
    #Perform tweet calculations
    calc.calc_followers_and_following()

    #Top 10 profiles overall by followers and following
    print("Top 10 followers - Overall\n")
    calc.profile_top(10,'followers')
    print("Top 10 following - Overall\n")
    calc.profile_top(10,'following')

    #Top 10 by profiles, per category, by followers and following
    for category in categories:
        print(f"Top 10 followers - {category} \n")
        calc.profile_top(10,'followers', account_category=category)
        print(f"Top 10 following - {category} \n")
        calc.profile_top(10,'following', account_category=category)
        
    #Horizontal Bar chart
    viz.hbar_tweets_by_col('account_category')
    
    
    #Hashtag Wordcloud
    #for category in categories:
    viz.wordcloud_hash_tags()

    #Tweet Wordlcloud
    for category in categories:
        viz.wordcloud_tweets(account_category=category)
        
    
    #Venn Diagram
    
    #Comparing Left Trolls to other categories
    category_list = ['LeftTroll', 'RightTroll']
    viz.venn_hashtags(category_list)
    
    category_list = ['LeftTroll', 'Fearmonger']
    viz.venn_hashtags(category_list)
    
    category_list = ['LeftTroll', 'Commercial']
    viz.venn_hashtags(category_list)
    
    category_list = ['LeftTroll', 'HashtagGamer']
    viz.venn_hashtags(category_list)
    
    category_list = ['LeftTroll', 'NewsFeed']
    viz.venn_hashtags(category_list)
    
    
    #Comparing Right Trolls to other categories
    category_list = ['RightTroll', 'Fearmonger']
    viz.venn_hashtags(category_list)
    
    category_list = ['RightTroll', 'Commercial']
    viz.venn_hashtags(category_list)
    
    category_list = ['RightTroll', 'HashtagGamer']
    viz.venn_hashtags(category_list)
    
    category_list = ['RightTroll', 'NewsFeed']
    viz.venn_hashtags(category_list)
    
    #Sentiment analysis donut chart
    for category in categories:
        viz.donut_sentiment(account_category=category)
        
if __name__ == '__main__':
    main()