import tweepy
import csv

# Choose variables for API keys:
consumer_key = 'XXX'
consumer_secret = 'XXX'
access_token = 'XXX'
access_token_secret = 'XXX'

#  Create API-Object
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

#  Choose name of the data file
csvFile = open('corona.csv', 'w+')
csvWriter = csv.writer(csvFile)
counter = 1

#  Collect tweets given keywords, ignore retweets, ignore tweets with links
#  Choose period of time, choose language, choose number of tweets (add here within parantheses: .items(NUM))
#  Create csv file with 4 columns: Number, date/time, location, tweet content/text

for tweet in tweepy.Cursor(api.search, q='(corona OR #corona OR covid-19 OR covid19 OR #covid19 OR sars-cov-2 OR #sarscov2 OR pandemic OR #pandemic) -filter:links -filter:retweets since:2020-05-15 until:2020-05-17', tweet_mode='extended', lang='en').items():
   csvWriter.writerow([counter, tweet.created_at, tweet.user.location, tweet.full_text])
   counter += 1


