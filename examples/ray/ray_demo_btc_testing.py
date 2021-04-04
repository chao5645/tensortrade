import ray
import numpy as np

from ray import tune
from ray.tune.registry import register_env

import tensortrade.env.default as default

from tensortrade.feed.core import DataFeed, Stream
from tensortrade.oms.instruments import Instrument
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Wallet, Portfolio

USD = Instrument("USD", 2, "U.S. Dollar")
TTC = Instrument("TTC", 8, "TensorTrade Coin")

import matplotlib.pyplot as plt

from tensortrade.env.generic import Renderer

import ta
import pandas as pd

from tensortrade.feed.core import Stream, DataFeed, NameSpace
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.instruments import USD, BTC
from tensortrade.oms.wallets import Wallet, Portfolio

def load_csv(filename):
    df = pd.read_csv('../data/' + filename, skiprows=0, index_col=0)
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
    print(df)

    return df

from tensortrade.env.default.renderers import PlotlyTradingChart

chart_renderer = PlotlyTradingChart(
    display=True,  # show the chart on screen (default)
    height=800,  # affects both displayed and saved file height. None for 100% height.
    save_format="html",  # save the chart to an HTML file
    auto_open_html=True,  # open the saved HTML chart in a new browser tab
)


def create_env(config):

    df = load_csv('btc_usdt_m5_history.csv')

    dataset = ta.add_trend_ta(df,'high', 'low', 'close', fillna=True)
    dataset = ta.add_volume_ta(dataset,'high', 'low', 'close', 'volume', fillna=True)
    #dataset = ta.add_volume_ta(dataset, 'high', 'low', 'close', 'volume', fillna=True)
    #dataset = ta.add_volatility_ta(dataset, 'high', 'low', 'close', fillna=True)
    price_history = dataset[['date', 'open', 'high', 'low', 'close', 'volume']]  # chart data
    dataset.drop(columns=['date', 'open', 'high', 'low', 'close', 'volume'], inplace=True)

    with NameSpace("bitfinex"):
        streams = [Stream.source(dataset[c].tolist(), dtype="float").rename(c) for c in dataset.columns]

    feed_ta_features = DataFeed(streams)



    price_list = price_history['close'].tolist()
    p = Stream.source(price_list, dtype="float").rename("USD-BTC")

    bitfinex = Exchange("bitfinex", service=execute_order)(
        p
    )

    cash = Wallet(bitfinex, 10000 * USD)
    asset = Wallet(bitfinex, 0 * BTC)

    portfolio = Portfolio(USD, [
        cash,
        asset
    ])

    reward_scheme = default.rewards.PBR(price=p)
    action_scheme = default.actions.BSHEX(
        cash=cash,
        asset=asset
    ).attach(reward_scheme)

    renderer_feed = DataFeed([
        Stream.source(price_list, dtype="float").rename("price"),
        Stream.sensor(action_scheme, lambda s: s.action, dtype="float").rename("action")
    ])

    renderer_feed_ptc = DataFeed([
        Stream.source(list(price_history["date"])).rename("date"),
        Stream.source(list(price_history["open"]), dtype="float").rename("open"),
        Stream.source(list(price_history["high"]), dtype="float").rename("high"),
        Stream.source(list(price_history["low"]), dtype="float").rename("low"),
        Stream.source(list(price_history["close"]), dtype="float").rename("close"),
        Stream.source(list(price_history["volume"]), dtype="float").rename("volume")
    ])

    env = default.create(
        feed=feed_ta_features,
        portfolio=portfolio,
        #renderer=PositionChangeChart(),
        renderer=default.renderers.PlotlyTradingChart(),
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        renderer_feed=renderer_feed_ptc,
        window_size=config["window_size"],
        max_allowed_loss=0.1
    )
    return env

register_env("TradingEnv", create_env)

import ray.rllib.agents.ppo as ppo
ray.init()
# Restore agent
agent = ppo.PPOTrainer(
    env="TradingEnv",
    config={
        "env_config": {
            "window_size": 25
        },
        "framework": "torch",
        "log_level": "DEBUG",
        "ignore_worker_failures": True,
        "num_workers": 1,
        "num_gpus": 0,
        "clip_rewards": True,
        "lr": 8e-6,
        "lr_schedule": [
            [0, 1e-1],
            [int(1e2), 1e-2],
            [int(1e3), 1e-3],
            [int(1e4), 1e-4],
            [int(1e5), 1e-5],
            [int(1e6), 1e-6],
            [int(1e7), 1e-7]
        ],
        "gamma": 0,
        "observation_filter": "MeanStdFilter",
        "lambda": 0.72,
        "vf_loss_coeff": 0.5,
        "entropy_coeff": 0.01
    }
)

checkpoint_path = "result\\PPO\\PPO_TradingEnv_54683_00000_0_2021-04-04_08-33-09\\checkpoint_100\\checkpoint-100"
agent.restore(checkpoint_path)

# Instantiate the environment
env_test = create_env({
    "window_size": 25
})

# Run until episode ends
episode_reward = 0
done = False
obs = env_test.reset()

while not done:
    action = agent.compute_action(obs)
    obs, reward, done, info = env_test.step(action)
    episode_reward += reward

    print("Info: ", info)

env_test.render()
