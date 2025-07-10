import sys
import os
import numpy as np
import pandas as pd
import matplotlib as plt
import yfinance as yf
import itertools

from packages.preprocess import prerpocess
from packages.agent import RLAgent
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

    # pd.set_option('display.max_rows', None)

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
    main()
