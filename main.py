import TweetVisualizer as viz
import TweetCalculator as calc

def main():
    instance = viz.TweetVisualizer()
    calculator = calc.TweetCalculator()
    
    cat_list = [None,'Fearmonger','Unknown','Commercial', 'HashtagGamer','LeftTroll', 'NewsFeed', 'RightTroll']
    calculator.calc_followers_and_following(instance.troll_tweet_df)
    instance.hbar_tweets_by_col('account_category')
    
    for category in cat_list:
        instance.wordcloud_hash_tags(account_category=category)
    
    for category in cat_list:
        instance.wordcloud_tweets(account_category=category)

main() 