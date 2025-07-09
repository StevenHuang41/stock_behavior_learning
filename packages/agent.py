import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from typing import Literal
import os

class RLAgent:
    def __init__(
        self, stock_no="0050.TW", n_TREND = 2, *,
        policy: Literal['q_learning', 'sarsa'],
        action_policy: Literal['epsilon_greedy', 'softmax_method'],
        alpha=0.001, gamma=0.9,
        epsilon=1, eps_dec=1e-3, eps_min=0.1,
        tau=1, tau_dec=1e-3, tau_min=0.3,
        episodes=1000,
    ):
        self.stock_no = stock_no
        self.n_TREND = n_TREND

        self.policy = policy
        self.action_policy = action_policy
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = float(eps_dec)
        self.eps_min = eps_min
        self.tau = tau
        self.tau_dec = float(tau_dec)
        self.tau_min = tau_min
        self.episodes = episodes

        self.TRENDS = ['up', 'down', 'stable']
        self.VOLUME_STATUS = ['high', 'low', 'normal']
        self.PORTFOLIO_STATUS = ['empty', 'holding']
        self.STATES = list(itertools.product(*[self.TRENDS] * n_TREND,
                                             self.VOLUME_STATUS,
                                             self.PORTFOLIO_STATUS))

        self.ACTIONS = ['buy', 'sell', 'hold']

        self.Q_table = {s: {a: 0 for a in self.ACTIONS} for s in self.STATES}

    def get_reward(
        self,
        price, previous_price, buy_price,
        state, action
    ) -> float:
        *_, pre_portfolio = state
     
        r = (price - previous_price) / previous_price
        if pre_portfolio == 'empty':
            if action == 'buy':
                if r > 0:
                    r *= 10
                
            elif action == 'sell':
                r = -1
            else : # action == 'hold'
                r = -r
                
        else : # pre_portfolio == 'holding'
            if action == 'buy':
                r = -1
            elif action == 'sell':
                if buy_price > 0:
                    r += (price - buy_price) / buy_price
                    # r *= 20
                else : # buy_price == 0
                    r = 0
            else : # action == 'hold'
                if buy_price > 0:
                    r += (price - buy_price) / buy_price


        return r

    def choose_action(self, action_policy, state, *, evaluate=False) -> str:
        if action_policy == 'epsilon_greedy':
            # set epsilon
            epsilon = self.epsilon if evaluate == False else 0

            if np.random.rand() < epsilon:
                # exploration
                return np.random.choice(self.ACTIONS)
            else :
                # exploitation
                return max(self.Q_table[state], key=self.Q_table[state].get)
        
        else : # action_policy == 'softmax_method'
            tau = self.tau if evaluate is False else 0.001

            q_values = np.fromiter(self.Q_table[state].values(), dtype=np.float64)
            exp_q_values = np.exp(q_values / tau)
            probs_actions = exp_q_values / np.sum(exp_q_values)
            
            return np.random.choice(self.ACTIONS, p=probs_actions)


    def update_Q_table(self, state, action, reward,
                       next_state=None, next_action=None, done=False):
        if self.policy == 'sarsa' and next_action is None and not done:
            raise ValueError("Must provide next action when using SARSA")

        if done:
            update_value = reward
        else :
            if self.policy == 'sarsa':
                best_next_action = next_action
            else : # policy == q_learning
                best_next_action = max(self.Q_table[next_state],
                                       key=self.Q_table[next_state].get)

            update_value = reward + \
            self.gamma * self.Q_table[next_state][best_next_action] \

        self.Q_table[state][action] += self.alpha * (
            update_value - self.Q_table[state][action]
        )

    def execute_action(
        action: str,
        portfolio: str,
        current_price: float,
        buy_price: float
    ) -> tuple[float, str]:



        print('in execute:')
        print(buy_price, portfolio)
        
        return buy_price, portfolio

    def train(self, df: pd.DataFrame):
        print(f"{self.policy} agent:")
        open_prices = df.iloc[:, 0].to_numpy()
        close_prices = df.iloc[:, 1].to_numpy()
        feature_cols = df.iloc[:, 2:].to_numpy()

        for episode in range(self.episodes):
            if episode % 100 == 0:
                if self.action_policy == 'epsilon_greedy':
                    print(f"    episode {episode:>4} progressing, epsilon {self.epsilon:.4f} ...")
                else :
                    print(f"    episode {episode:>4} progressing, tau {self.tau:.4f} ...")
                
            ## Initialize protfolio status
            portfolio = 'empty'
            buy_price = 0

            # Initial state
            state = (*feature_cols[0], portfolio)
            action = self.choose_action(self.action_policy, state)

            # go through the whole df rows
            for i in range(1, len(df) - 1):
                previous_price = close_prices[i - 1]
                price = open_prices[i]

                ## calculate reward
                reward = self.get_reward(
                    price, previous_price, buy_price, state, action
                )

                ## execute action
                if action == 'buy' and portfolio == 'empty':
                    buy_price = price
                    portfolio = 'holding'
                elif action == 'sell' and portfolio == 'holding':
                    buy_price = 0
                    portfolio = 'empty'

                ## get new state
                next_state = (*feature_cols[i], portfolio)

                ## set next action 
                if self.policy == 'q_learning':
                    next_action = None
                else : # sarsa
                    next_action = self.choose_action(self.action_policy, next_state)

                ## update Q table
                self.update_Q_table(state, action, reward, next_state, next_action)

                ## move to next state and action
                state = next_state
                if self.policy == 'sarsa':
                    action = next_action
                else : # q-learning
                    action = self.choose_action(self.action_policy, state)

            ## last record
            previous_price = close_prices[-2]
            price = open_prices[-1]

            reward = self.get_reward(price, previous_price, buy_price,
                                     state, action)
            
            self.update_Q_table(state, action, reward, done=True)

            ###
            ## version 1: fast drop at the beginning
            # self.epsilon = self.epsilon * (1 - self.eps_dec) \
            #     if self.epsilon > self.eps_min \
            #     else self.eps_min
            ## version 2: gradually drop
            if self.action_policy == 'epsilon_greedy':
                self.epsilon = self.epsilon - self.eps_dec \
                                if self.epsilon > self.eps_min \
                                else self.eps_min
            else : # action policy = softmax_method
                self.tau = self.tau - self.tau_dec \
                            if self.tau > self.tau_min \
                            else self.tau_min
            

        print("=" * 10, "Finished Training", "=" * 10)
            
    def evaluate_learning(self, df: pd.DataFrame, initial_cash=10000):
        ## Initial status
        # traditional strategy
        cash_tra = initial_cash
        shares_tra = 0
        values_tra = []

        # learning strategy
        cash = initial_cash
        shares = 0
        portfolio = 'empty'
        values_learning = []
        
        open_prices = df.iloc[:, 0].to_numpy()
        close_prices = df.iloc[:, 1].to_numpy()
        feature_cols = df.iloc[:, 2:].to_numpy()

        # traditional holds from the start
        shares_tra = cash_tra // open_prices[0]
        cash_tra -= shares_tra * open_prices[0]
        values_tra.append(cash_tra + shares_tra * close_prices[0])

        # learning check for day1 state then choose action
        state = (*feature_cols[0], portfolio)
        action = self.choose_action(self.action_policy, state, evaluate=True)
        values_learning.append(initial_cash) # did not do anything at the first day

        for i in range(1, len(df)):
            opening_p = open_prices[i]
            closing_p = close_prices[i]

            ## execute action
            if action == 'buy' and portfolio == 'empty':
                if cash // opening_p > 0:
                    shares = cash // opening_p
                    cash -= shares * opening_p
                    portfolio = 'holding'
            elif action == 'sell' and portfolio == 'holding':
                if shares > 0:
                    cash += shares * opening_p
                    shares = 0
                    portfolio = 'empty'

            values_tra.append(cash_tra + shares_tra * closing_p)
            values_learning.append(cash + shares * closing_p)

            # set next state
            state = (*feature_cols[i], portfolio)
            action = self.choose_action(self.action_policy, state, evaluate=True)

        ## plot fig
        plt.figure(constrained_layout=True)
        plt.plot(values_tra, label="holding", alpha=0.3, linestyle='--')
        plt.plot(values_learning, label=f"{self.policy} {self.action_policy}")
        plt.xlabel('Holding Days')
        plt.ylabel('Portfolio value')
        plt.title(f"{self.stock_no}")

        apc = 100 * (values_learning[-1] - initial_cash) / initial_cash
        hpc = 100 * (values_tra[-1] - initial_cash) / initial_cash
        cpc = 100 * (values_learning[-1] - values_tra[-1]) / values_tra[-1]

        apc_c = 'green' if apc >= 0 else 'red'
        hpc_c = 'green' if hpc >= 0 else 'red'
        cpc_c = 'green' if cpc >= 0 else 'red'

        text_diff = 3000 if cpc >= 0 else -3000

        plt.annotate(
            f"Agent: {values_learning[-1]:.0f}",
            xy=(len(values_learning), values_learning[-1]),
            xytext=(len(values_learning) + 500, values_learning[-1] + text_diff),
            fontsize=12,
            ha='left',
            va='center',
            arrowprops={
                'arrowstyle': "->",
                'color': 'orange',
                'alpha': 0.5,
            }
        )

        plt.annotate(
            f"Holding: {values_tra[-1]:.0f}",
            xy=(len(values_tra), values_tra[-1]),
            xytext=(len(values_tra) + 500, values_tra[-1] - text_diff),
            fontsize=12,
            ha='left',
            va='center',
            arrowprops={
                'arrowstyle': "->",
                'color': 'blue',
                'alpha': 0.3,
            }
        )

        plt.figtext(1.2, 0.8, 'Cash Percentage Change', fontsize=12)
        plt.figtext(1.25, 0.75, f'agent growth:    {apc:^+9.2f} %', fontsize=12, color=apc_c)
        plt.figtext(1.25, 0.7,  f'holding growth:  {hpc:^+9.2f} %', fontsize=12, color=hpc_c)
        plt.figtext(1.25, 0.65, f'relative change: {cpc:^+9.2f} %', fontsize=12, color=cpc_c)
        plt.legend()
        # plt.tight_layout()

        ## save fig
        images_dir = os.path.join(os.getcwd(), 'images')
        os.makedirs(images_dir, exist_ok=True)
        fig_fname = f'{self.policy}_{self.action_policy}.png'
        plt.savefig(os.path.join(images_dir, fig_fname), bbox_inches='tight')

        ## show fig
        plt.show()

        print(f"Final value traditional strategy:\n \
                cash={cash_tra}\n \
                shares={shares_tra * close_prices[-1]}\n \
                Value={values_tra[-1]}")
        print(f"Final value {self.policy}:\n \
                cash={cash}\n \
                shares={shares * close_prices[-1]}\n \
                Value={values_learning[-1]}")

def show_q_table(agent):
    for k in agent.Q_table.keys():
        best_action = max(agent.Q_table[k], key=agent.Q_table[k].get)
        print(f"{str(k):<50} -> {best_action:<8}", end='')
        for i, v in agent.Q_table[k].items():
            print(f"{i:<4}: {v:+.4f}", end='  ')

        print()

if __name__ == "__main__":
    q_epsilon_agent = RLAgent(
        policy='q_learning',
        action_policy='epsilon_greedy',
        gamma=0.9,
    )

    from preprocess import prerpocess
    import yfinance as yf

    stock_data = yf.download("0050.TW", period="max", auto_adjust=True)

    stock_data = prerpocess(stock_data,
                            hasStockSplited=True,
                            split_date='2025-06-06',
                            split_ratio=4)

    q_epsilon_agent.train(stock_data)
    q_epsilon_agent.evaluate_learning(stock_data)

    show_q_table(q_epsilon_agent)