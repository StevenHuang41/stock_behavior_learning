import sys
import os
import numpy as np
import pandas as pd
import matplotlib as plt
import yfinance as yf
import itertools

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

from packages.preprocess import prerpocess, deep_agent_preprocess
from packages.agent import RLAgent
from packages.deep_learning_agent import DQNAgent, DsarsaAgent
from packages.deep_learning_agent import EpsilonGreedy, SoftmaxMethod

# note: 2025/6/6-9 stock split

# states
TRENDS = ['up', 'down', 'stable']
VOLUME_STATUS = ['high', 'low', 'normal']
PORTFOLIO_STATUS = ['empty', 'holding']
STATES = list(itertools.product(*[TRENDS] * 2, VOLUME_STATUS, PORTFOLIO_STATUS))

ACTIONS = ['buy', 'sell', 'hold']

def main():

    # get stock number
    if len(sys.argv) == 2:
        stock_no = sys.argv[1]

    elif len(sys.argv) == 4:
        splited = True
        stock_no = (sys.argv[1]).strip()
        split_date = (sys.argv[2]).strip()
        split_ratio = float((sys.argv[3]).strip())
    else :
        stock_no = input("Enter a Stock number:\n")
        splited = input(f"Has {stock_no} splited?\n")
        splited = splited.strip().lower()
        splited = True if 'y' in splited else False
        if splited:
            split_date = input("When did it split? ['YYYY-MM-DD']\n").strip()
            split_ratio = float(input("What is the split ratio? (n-for-1)\n").strip())

    # download stock data from yf
    stock_data = yf.download(stock_no, period="max", auto_adjust=True)

    # do preprocessing
    stock_data = prerpocess(stock_data,
                            hasStockSplited=splited,
                            split_date=split_date,
                            split_ratio=split_ratio)


    ## TODO use skopt to tune parameters
    # search_space = [
    #     Real(0.0001, 1.0, name='alpha'),
    #     Real(0.8, 0.99, name='gamma')
    # ]
    # @use_named_args(search_space)
    # def tuning_agent(alpha, gamma):
    #     q_epsilon_agent = RLAgent(
    #         policy='q_learning',
    #         action_policy='epsilon_greedy',
    #         alpha=alpha, gamma=gamma,
    #     )
    #     q_epsilon_agent.train(stock_data)
    #     score = q_epsilon_agent.evaluate_learning(stock_data)
    #     return -score # look for minimum value result
    # result = gp_minimize(
    #     func=tuning_agent,
    #     dimensions=search_space,
    #     n_calls=30,
    # )
    # print(result.x)

    # # agent 1: q learning, epsilon greedy
    # q_epsilon_agent = RLAgent(
    #     policy='q_learning',
    #     action_policy='epsilon_greedy',
    #     alpha=0.001, gamma=0.9,
    # )
    # q_epsilon_agent.train(stock_data)

    # # agent 2: q learning, softmax method
    # q_soft_agent = RLAgent(
    #     policy='q_learning',
    #     action_policy='softmax_method',
    #     alpha=0.1, gamma=0.9,
    # )
    # q_soft_agent.train(stock_data)

    # agent 3: sarsa, epsilon greed 
    # s_epsilon_agent = RLAgent(
    #     policy='sarsa',
    #     action_policy='epsilon_greedy',
    #     alpha=0.001, gamma=0.9,
    # )
    # s_epsilon_agent.train(stock_data)

    # agent 4: sarsa, softmax method
    # s_soft_agent = RLAgent(
    #     policy='sarsa',
    #     action_policy='softmax_method',
    #     alpha=0.001, gamma=0.9,
    # )
    # s_soft_agent.train(stock_data)

    stock_data = deep_agent_preprocess(stock_data)
    # exclude Open and Close price, but add one for portfolio
    state_size = len(stock_data.columns) - 2 + 1

    # agent 5: Deep q learning, epsilon greedy
    # dqn_eps_agent = DQNAgent(
    #     action_policy=EpsilonGreedy(),
    #     state_size=state_size,
    #     action_size=len(ACTIONS),
    #     alpha=0.001, gamma=0.1,
    #     episodes=100,
    #     apn='epsilon_greedy'
    # )
    # dqn_eps_agent.initialize()
    # dqn_eps_agent.train(stock_data)
    # dqn_eps_agent.evaluate_learning(stock_data)
    # dqn_eps_agent.show_performance()

    # agent 6: Deep q learning, softmax method
    dqn_soft_agent = DQNAgent(
        action_policy=SoftmaxMethod(),
        state_size=state_size,
        action_size=len(ACTIONS),
        alpha=0.001, gamma=0.9,
        episodes=100,
        apn='softmax_method'
    )
    dqn_soft_agent.initialize()
    dqn_soft_agent.train(stock_data)
    dqn_soft_agent.evaluate_learning(stock_data)
    dqn_soft_agent.show_performance()




#####################################################################################
    # # agent 1
    # q_epsilon_agent.evaluate_learning(stock_data)
    # q_epsilon_agent.store_q_table()

    # # agent 2
    # q_soft_agent.evaluate_learning(stock_data)
    # q_soft_agent.store_q_table()

    # # agent 3
    # s_epsilon_agent.evaluate_learning(stock_data)
    # s_epsilon_agent.store_q_table()

    # # agent 4
    # s_soft_agent.evaluate_learning(stock_data)
    # s_soft_agent.store_q_table()





if __name__ == "__main__":

    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)

    main()
