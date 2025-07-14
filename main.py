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


def show_usage():
    print("Usage: python main.py [stock no]\n\n"
          "    stock no: should be able to be found using yfinance\n")

def main():
    # get stock number
    if len(sys.argv) == 1:
        stock_no = (input("Enter a Stock number:\n")).strip()

    elif len(sys.argv) == 2:
        if sys.argv[1] in ['-h', '--help']:
            show_usage()
            sys.exit(0)

        stock_no = (sys.argv[1]).strip()

    else :
        show_usage()
        sys.exit(1)

    stock_data = yf.download(stock_no, period="max", auto_adjust=True)

    if stock_data.empty:
        print(f'\nError: No data found for stock no: {stock_no}.',
              'Please check the stock no.')
        sys.exit(1)

    # do preprocessing
    avg_days = [5, 20] # set average of 5, 20 days preiods

    # states
    # TRENDS = ['up', 'down', 'stable']
    # VOLUME_STATUS = ['high', 'low', 'normal']
    # PORTFOLIO_STATUS = ['empty', 'holding']

    ACTIONS = ['buy', 'sell', 'hold']
    # STATES = list(itertools.product(*[TRENDS] * len(avg_days),
                                    # VOLUME_STATUS,
                                    # PORTFOLIO_STATUS))

    stock_data = prerpocess(stock_no, stock_data, avg_days)

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

    # agent 1: q learning, epsilon greedy
    q_epsilon_agent = RLAgent(
        policy='q_learning',
        action_policy='epsilon_greedy',
        alpha=0.001, gamma=0.9,
    )
    q_epsilon_agent.train(stock_data)

    # agent 2: q learning, softmax method
    q_soft_agent = RLAgent(
        policy='q_learning',
        action_policy='softmax_method',
        alpha=0.1, gamma=0.9,
    )
    q_soft_agent.train(stock_data)

    # agent 3: sarsa, epsilon greed 
    s_epsilon_agent = RLAgent(
        policy='sarsa',
        action_policy='epsilon_greedy',
        alpha=0.001, gamma=0.9,
    )
    s_epsilon_agent.train(stock_data)

    # agent 4: sarsa, softmax method
    s_soft_agent = RLAgent(
        policy='sarsa',
        action_policy='softmax_method',
        alpha=0.001, gamma=0.9,
    )
    s_soft_agent.train(stock_data)

#######################################################################
    stock_data = deep_agent_preprocess(stock_data)

    # agent 5: Deep q learning, epsilon greedy
    dqn_eps_agent = DQNAgent(
        action_policy=EpsilonGreedy(),
        apn='epsilon_greedy',
        alpha=0.001,
        gamma=0.9,
        episodes=100,
    )
    dqn_eps_agent.initialize()
    dqn_eps_agent.train(stock_data)

    # agent 6: Deep q learning, softmax method
    dqn_soft_agent = DQNAgent(
        action_policy=SoftmaxMethod(),
        apn='softmax_method',
        alpha=0.001,
        gamma=0.9,
        episodes=100,
    )
    dqn_soft_agent.initialize()
    dqn_soft_agent.train(stock_data)

    # agent 7: Deep sarsa learning, epsilon greedy
    ds_eps_agent = DsarsaAgent(
        action_policy=EpsilonGreedy(),
        apn='epsilon_greedy',
        alpha=0.001,
        gamma=0.8,
        episodes=100,
    )
    ds_eps_agent.initialize()
    ds_eps_agent.train(stock_data)

    # agent 8: Deep sarsa learning, softmax method
    ds_soft_agent = DsarsaAgent(
        action_policy=SoftmaxMethod(),
        apn='softmax_method',
        alpha=0.001,
        gamma=0.8,
        episodes=100,
    )
    ds_soft_agent.initialize()
    ds_soft_agent.train(stock_data)

##########################################################################

    # agent 1
    q_epsilon_agent.evaluate_learning(stock_data)
    q_epsilon_agent.store_q_table()

    # agent 2
    q_soft_agent.evaluate_learning(stock_data)
    q_soft_agent.store_q_table()

    # agent 3
    s_epsilon_agent.evaluate_learning(stock_data)
    s_epsilon_agent.store_q_table()

    # agent 4
    s_soft_agent.evaluate_learning(stock_data)
    s_soft_agent.store_q_table()

    # agent 5
    dqn_eps_agent.evaluate_learning(stock_data)
    dqn_eps_agent.show_performance()

    # agent 6
    dqn_soft_agent.evaluate_learning(stock_data)
    dqn_soft_agent.show_performance()

    # agent 7
    ds_eps_agent.evaluate_learning(stock_data)
    ds_eps_agent.show_performance()

    # agent 8
    ds_soft_agent.evaluate_learning(stock_data)
    ds_soft_agent.show_performance()



if __name__ == "__main__":

    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)

    main()
