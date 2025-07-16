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

# note: 2025/6/6 0050.TW stock split

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

    # preprocessing
    avg_days = [5, 20] # set average of 5, 20 days preiods

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

    """
    Usage Example for classic RL agent:
        agent = RLAgent(
            stock_no=stock_no,
            len_avg_days=len(avg_days),
            policy= ['q_learning' , 'sarsa'],
            action_policy=['epsilon_greedy' , 'softmax_method'],
            alpha=[0.001],
            gamma=[0.9],
            episodes=[1000],
        )

        agent.train(stock_data)
        # or use the best recorded performance 
        # agent.load_q_table(load_best=True)

        # save image
        agent.evaluate_learning(stock_data)

        # save document
        agent.write_document()


    Usage Example for Deep Q-Network agent:
        deep_agent = [DQNAgent | DsarsaAgent](
            action_policy= [EpsilonGreedy()| ,SoftmaxMethod()]
            apn= ['epsilon_greedy', 'softmax_method'] # should match the above action policy
            alpha=[0.001],
            gamma=[0.9],
            episodes=[100],
        )
        deep_agent.train(stock_data_) # use 'stock_data_' for deep agent learning
        # or use the best recorded performance 
        deep_agent.load_weight(load_best=True)

        # save image
        deep_agent.evaluate_learning(stock_data_)
        
        # save document
        deep_agent.write_document()


    Adjust the parameters as you may, you could try to produce the best performance,
    and see the result in images/ or document/ after evaluate_learning().

    """
    # agent 1: q learning, epsilon greedy
    q_epsilon_agent = RLAgent(
        stock_no=stock_no,
        len_avg_days=len(avg_days),
        policy='q_learning',
        action_policy='epsilon_greedy',
        alpha=0.001,
        gamma=0.9,
        episodes=1000,
    )
    q_epsilon_agent.train(stock_data)
    # q_epsilon_agent.load_q_table(load_best=True)
    q_epsilon_agent.evaluate_learning(stock_data)
    q_epsilon_agent.write_document()

    # agent 2: q learning, softmax method
    q_soft_agent = RLAgent(
        stock_no=stock_no,
        len_avg_days=len(avg_days),
        policy='q_learning',
        action_policy='softmax_method',
        alpha=0.1,
        gamma=0.9,
        episodes=1000,
    )
    q_soft_agent.train(stock_data)
    # q_soft_agent.load_q_table(load_best=True)
    q_soft_agent.evaluate_learning(stock_data)
    q_soft_agent.write_document()

    # agent 3: sarsa, epsilon greed 
    s_epsilon_agent = RLAgent(
        stock_no=stock_no,
        len_avg_days=len(avg_days),
        policy='sarsa',
        action_policy='epsilon_greedy',
        alpha=0.001,
        gamma=0.9,
        episodes=1000,
    )
    s_epsilon_agent.train(stock_data)
    # s_epsilon_agent.load_q_table(load_best=True)
    s_epsilon_agent.evaluate_learning(stock_data)
    s_epsilon_agent.write_document()

    # agent 4: sarsa, softmax method
    s_soft_agent = RLAgent(
        stock_no=stock_no,
        len_avg_days=len(avg_days),
        policy='sarsa',
        action_policy='softmax_method',
        alpha=0.001,
        gamma=0.9,
        episodes=1000,
    )
    s_soft_agent.train(stock_data)
    # s_soft_agent.load_q_table(load_best=True)
    s_soft_agent.evaluate_learning(stock_data)
    s_soft_agent.write_document()

#######################################################################
    stock_data_ = deep_agent_preprocess(stock_data)

    # agent 5: Deep q learning, epsilon greedy
    dqn_eps_agent = DQNAgent(
        stock_no="0050.TW",
        len_avg_days=len(avg_days),
        action_policy=EpsilonGreedy(),
        apn='epsilon_greedy',
        alpha=0.001,
        gamma=0.9,
        episodes=10,
    )
    dqn_eps_agent.train(stock_data_)
    # dqn_eps_agent.load_weight(load_best=True)
    dqn_eps_agent.evaluate_learning(stock_data_)
    dqn_eps_agent.write_document()

    # agent 6: Deep q learning, softmax method
    dqn_soft_agent = DQNAgent(
        stock_no="0050.TW",
        len_avg_days=len(avg_days),
        action_policy=SoftmaxMethod(),
        apn='softmax_method',
        alpha=0.001,
        gamma=0.9,
        episodes=10,
    )
    dqn_soft_agent.train(stock_data_)
    # dqn_soft_agent.load_weight(load_best=True)
    dqn_soft_agent.evaluate_learning(stock_data_)
    dqn_soft_agent.write_document()

    # agent 7: Deep sarsa learning, epsilon greedy
    ds_eps_agent = DsarsaAgent(
        stock_no="0050.TW",
        len_avg_days=len(avg_days),
        action_policy=EpsilonGreedy(),
        apn='epsilon_greedy',
        alpha=0.001,
        gamma=0.8,
        episodes=10,
    )
    ds_eps_agent.train(stock_data_)
    # ds_eps_agent.load_weight(load_best=True)
    ds_eps_agent.evaluate_learning(stock_data_)
    ds_eps_agent.write_document()

    # agent 8: Deep sarsa learning, softmax method
    ds_soft_agent = DsarsaAgent(
        stock_no="0050.TW",
        len_avg_days=len(avg_days),
        action_policy=SoftmaxMethod(),
        apn='softmax_method',
        alpha=0.001,
        gamma=0.8,
        episodes=10,
    )
    ds_soft_agent.train(stock_data_)
    # ds_soft_agent.load_weight(load_best=True)
    ds_soft_agent.evaluate_learning(stock_data_)
    ds_soft_agent.write_document()




if __name__ == "__main__":
    main()
