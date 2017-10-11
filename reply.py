#!/usr/bin/env python
import tweepy
from tweepy.streaming import StreamListener

import webai as w

from credentials import *




# Access and authorize our Twitter credentials from credentials.py
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

B="@realDonaldTrump"
new_tweets = api.user_timeline(screen_name =B,count=200)

# list of specific strings we want to check for in Tweets



for s in new_tweets:

            sn = s.user.screen_name
            ai=w.rundis(s.text)
            m = " @"+sn+ ai
            if len(m) > 140:
                m = m[0:139] + ' '

            s = api.update_status(m, s.id)

