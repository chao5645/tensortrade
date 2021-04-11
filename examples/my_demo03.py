from abc import abstractmethod

import numpy as np
import pandas as pd

from tensortrade.env.generic import RewardScheme, TradingEnv
from tensortrade.feed.core import Stream, DataFeed
import math


def load_csv(filename):
    df = pd.read_csv('data/' + filename, skiprows=0, index_col=0)
    #df.drop(columns=['symbol', 'volume_btc'], inplace=True)

    # Fix timestamp form "2019-10-17 09-AM" to "2019-10-17 09-00-00 AM"
    #df['date'] = df['date'].str[:14] + '00-00 ' + df['date'].str[-2:]

    # Convert the date column type from string to datetime for proper sorting.
    df['date'] = pd.to_datetime(df['date'])

    # Make sure historical prices are sorted chronologically, oldest first.
    df.sort_values(by='date', ascending=True, inplace=True)

    df.reset_index(drop=True, inplace=True)

    # Format timestamps as you want them to appear on the chart buy/sell marks.
    df['date'] = df['date'].dt.strftime('%Y-%m-%d %I:%M %p')

    return df

df = load_csv('btc_usdt_m5_history.csv')
price = df['close']

position = 1.1

r = Stream.sensor(price, lambda p: p.value, dtype="float").diff()
position = Stream.sensor(position, lambda p: p, dtype="float")

reward = (position * r).fillna(0).rename("reward")

print(r)

print("Type: {}".format(type(r)))

print("--------------------------")
print(position)

print("Type: {}".format(type(position)))

print("--------------------------")
print(reward)
#print(reward.forward())

print("Type: {}".format(type(reward)))

#feed = DataFeed([reward])
#feed.compile()