
import ta

import pandas as pd

from tensortrade.feed.core import Stream, DataFeed, NameSpace
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.instruments import USD, BTC
from tensortrade.oms.wallets import Wallet, Portfolio


#%%

def load_csv(filename):
    df = pd.read_csv('data/' + filename, skiprows=1)
    df.drop(columns=['symbol', 'volume_btc'], inplace=True)

    # Fix timestamp form "2019-10-17 09-AM" to "2019-10-17 09-00-00 AM"
    df['date'] = df['date'].str[:14] + '00-00 ' + df['date'].str[-2:]

    # Convert the date column type from string to datetime for proper sorting.
    df['date'] = pd.to_datetime(df['date'])

    # Make sure historical prices are sorted chronologically, oldest first.
    df.sort_values(by='date', ascending=True, inplace=True)

    df.reset_index(drop=True, inplace=True)

    # Format timestamps as you want them to appear on the chart buy/sell marks.
    df['date'] = df['date'].dt.strftime('%Y-%m-%d %I:%M %p')

    return df

#%%
df = load_csv('Coinbase_BTCUSD_1h.csv')
dataset = ta.add_all_ta_features(df, 'open', 'high', 'low', 'close', 'volume', fillna=True)
price_history = dataset[['date', 'open', 'high', 'low', 'close', 'volume']]  # chart data

dataset.drop(columns=['date', 'open', 'high', 'low', 'close', 'volume'], inplace=True)




bitfinex = Exchange("bitfinex", service=execute_order)(
    Stream.source(price_history['close'].tolist(), dtype="float").rename("USD-BTC")
)

portfolio = Portfolio(USD, [
    Wallet(bitfinex, 10000 * USD),
    Wallet(bitfinex, 0 * BTC),
])

with NameSpace("bitfinex"):
    streams = [Stream.source(dataset[c].tolist(), dtype="float").rename(c) for c in dataset.columns]

feed = DataFeed(streams)
print("--------Feed Next--------")
print(feed.next())
print("-------------------------")


from tensortrade.env.default.renderers import PlotlyTradingChart, FileLogger

chart_renderer = PlotlyTradingChart(
    display=True,  # show the chart on screen (default)
    height=800,  # affects both displayed and saved file height. None for 100% height.
    save_format="html",  # save the chart to an HTML file
    auto_open_html=True,  # open the saved HTML chart in a new browser tab
)

file_logger = FileLogger(
    filename="example.log",  # omit or None for automatic file name
    path="training_logs"  # create a new directory if doesn't exist, None for no directory
)



import tensortrade.env.default as default
renderer_feed = DataFeed([
    Stream.source(price_history[c].tolist(), dtype="float").rename(c) for c in price_history]
)
from tensortrade.env.default.actions import ManagedRiskOrders
from tensortrade.env.default.rewards import RiskAdjustedReturns

action = ManagedRiskOrders(
    stop=[0.05, 0.10, 0.15],
    take=[0.10, 0.20, 0.50],

)

reward  = RiskAdjustedReturns(
    window_size=10
)
env = default.create(
    portfolio=portfolio,
    action_scheme="managed-risk",
    reward_scheme="risk-adjusted",
    feed=feed,
    window_size=20,
    renderer_feed=renderer_feed,
    renderer=chart_renderer

)


print("--------action_scheme--------")
print(env.action_scheme.action_space)



from tensortrade.agents import DQNAgent, A2CAgent

agent = A2CAgent(env)

# Set render_interval to None to render at episode ends only
agent.train(n_episodes=2, n_steps=1000, render_interval=500, save_path="agents/")

#agent.policy_network.save(filepath="agents02")
agent.save(path="agents0023")

#agent.restore(path="agents0023policy_network__d538fde__20210310_231131.hdf5")

# Run until episode ends
episode_reward = 0
done = True
#obs = env.reset()

while not done:

    #print("Obs:", obs)
    action = agent.get_action(obs)
    #print("Action: ", action)
    obs, reward, done, info = env.step(action)
    episode_reward += reward
    #print("episode_reward: ", episode_reward)
env.render()
