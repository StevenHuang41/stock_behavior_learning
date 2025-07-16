import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from typing import Literal
import os
import sys
from tqdm import tqdm
import pickle

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

        self.values_learning_last = None

        self.ACTIONS = ['buy', 'sell', 'hold']

        self.Q_table = {s: {a: 0 for a in self.ACTIONS} for s in self.STATES}

    def _save_q_table(self):
        os.makedirs('model_weights', 0o755, exist_ok=True)
        with open(f'model_weights/{self.policy}_{self.action_policy}.pkl', 'wb') as f:
            pickle.dump(self.Q_table, f)

    def load_q_table(self, *, load_best=False):
        if load_best and not os.path.exists('best_performance'):
            return FileNotFoundError('best_performance has not been created.')

        store_dir = 'best_performance' if load_best else 'model_weights'
        file_name = f'{self.policy}_{self.action_policy}.pkl'

        if os.path.isfile(f'{store_dir}/{file_name}'):
            with open(f'{store_dir}/{file_name}', 'rb') as f:
                self.Q_table = pickle.load(f)
        else :
            raise FileNotFoundError(f'{file_name} does not exist.')

    def _get_reward(
        self,
        price, previous_price, buy_price,
        pre_portfolio, action
    ) -> float:

        r = (price - previous_price) / previous_price
        if pre_portfolio == 'empty':
            if action == 'buy':
                r = 0

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
                else :
                    r = 0

            else : # action == 'hold'
                if buy_price > 0:
                    r += (price - buy_price) / buy_price
                else :
                    r = 0
                    
        return r

    def _choose_action(self, state, *, evaluate=False) -> str:
        if self.action_policy == 'epsilon_greedy':
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
            q_values_div_tau = q_values / tau
            q_values_div_tau = q_values_div_tau - np.max(q_values_div_tau) # for stability
            exp_q_values = np.exp(q_values_div_tau)
            probs_actions = exp_q_values / np.sum(exp_q_values)
            
            return np.random.choice(self.ACTIONS, p=probs_actions)


    def _update_Q_table(self, state, action, reward,
                       next_state, next_action, done=False):
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
            self.gamma * self.Q_table[next_state][best_next_action]

        self.Q_table[state][action] += self.alpha * (
            update_value - self.Q_table[state][action]
        )

    def train(self, df: pd.DataFrame):
        print(f"{self.policy} {self.action_policy} agent:")
        open_prices = df.iloc[:, 0].to_numpy()
        close_prices = df.iloc[:, 1].to_numpy()
        feature_cols = df.iloc[:, 2:].to_numpy()

        pbar = tqdm(range(self.episodes), ncols=100)
        for episode in pbar:
            pbar.set_description(f"Episode: {episode + 1}")

            ## Initialize protfolio status
            portfolio = 'empty'
            buy_price = 0

            # Initial state
            state = (*feature_cols[0], portfolio)
            action = self._choose_action(state)

            # go through the whole df rows
            for i in range(1, len(df)):
                previous_price = close_prices[i - 1]
                price = open_prices[i]

                ## calculate reward
                reward = self._get_reward(
                    price, previous_price, buy_price, portfolio, action
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
                next_action = self._choose_action(next_state)

                done = False if i != len(df) - 1 else True

                ## update Q table
                self._update_Q_table(state, action, reward,
                                    next_state, next_action, done)

                ## move to next state and action
                state = next_state
                action = next_action

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
        
  
            
    def evaluate_learning(self, df: pd.DataFrame, initial_cash=10000) -> float:
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
        action = self._choose_action(state, evaluate=True)
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
            action = self._choose_action(state, evaluate=True)

        ## plot fig
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.subplots_adjust(right=0.55)

        ax.plot(values_tra, label="holding", alpha=0.3, linestyle='--')
        ax.plot(values_learning, label=f"{self.policy} {self.action_policy}")
        ax.plot([], [], ' ', label=f'alpha={self.alpha}\ngamma={self.gamma}')
        ax.set_xlabel('Holding Days')
        ax.set_ylabel('Portfolio value')
        ax.set_title(f"{self.stock_no}")

        apc = 100 * (values_learning[-1] - initial_cash) / initial_cash
        hpc = 100 * (values_tra[-1] - initial_cash) / initial_cash
        cpc = 100 * (values_learning[-1] - values_tra[-1]) / values_tra[-1]

        apc_c = 'green' if apc >= 0 else 'red'
        hpc_c = 'green' if hpc >= 0 else 'red'
        cpc_c = 'green' if cpc >= 0 else 'red'

        text_diff = 3000 if cpc >= 0 else -3000

        ax.annotate(
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

        ax.annotate(
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

        fig.text(0.74, 0.8,
                 'Cash Percentage Change', fontsize=12)
        fig.text(0.75, 0.75,
                 f'agent growth:    {apc:^+9.2f} %', fontsize=12, color=apc_c)
        fig.text(0.75, 0.71,
                 f'holding growth:  {hpc:^+9.2f} %', fontsize=12, color=hpc_c)
        fig.text(0.75, 0.67,
                 f'relative change: {cpc:^+9.2f} %', fontsize=12, color=cpc_c)

        self.today_date = str(df.index[-1]).split(' ')[0]
        empty_state = (*df.iloc[-1, 2:].to_numpy(), 'empty')
        holding_state = (*df.iloc[-1, 2:].to_numpy(), 'holding')
        self.empty_action = self._choose_action(empty_state, evaluate=True)
        self.holding_action = self._choose_action(holding_state, evaluate=True)

        fig.text(0.74, 0.5,  f'Today Date : {self.today_date}', fontsize=12)
        fig.text(0.74, 0.45, f'Next Action:', fontsize=12)
        fig.text(0.75, 0.41, f'portfolio holding —> {self.holding_action}', fontsize=12)
        fig.text(0.75, 0.37, f'portfolio empty   —> {self.empty_action}', fontsize=12)

        ax.legend()

        ## save fig
        images_dir = os.path.join(os.getcwd(), 'images')
        os.makedirs(images_dir, exist_ok=True)
        fig_fname = f'{self.policy}_{self.action_policy}.png'
        fig.savefig(os.path.join(images_dir, fig_fname), bbox_inches='tight')

        ## show fig
        # plt.show()

        ## store in self
        self.cash_tra = cash_tra
        self.shares_tra = shares_tra * close_prices[-1]
        self.values_tra_last = values_tra[-1]
        self.cash = cash
        self.shares = shares * close_prices[-1]
        self.values_learning_last = values_learning[-1]

        self._save_q_table()

        return values_learning[-1]

        

    def write_document(self):
        if self.values_learning_last == None:
            raise RuntimeError("Run write_document() after evaluate_learning().")

        original_stdout = sys.stdout
        os.makedirs('documents', mode=0o755, exist_ok=True)
        with open(f'documents/{self.policy}_{self.action_policy}.txt', 'w') as f:
            sys.stdout = f

            print(f"Final value traditional strategy:\n"
                  f"    cash={self.cash_tra}\n"
                  f"    shares={self.shares_tra}\n"
                  f"    Value={self.values_tra_last}")

            print(f"\nFinal value {self.policy}:\n"
                  f"    cash={self.cash}\n"
                  f"    shares={self.shares}\n"
                  f"    Value={self.values_learning_last}\n")

            print(f'{'States':<45}|{'Best Action':<15}| q values')
            print('-' * 110)
            for k in self.Q_table.keys():
                best_action = max(self.Q_table[k], key=self.Q_table[k].get)
                print(f"{str(k):<45}|{best_action:<15}| ", end='')
                for i, v in self.Q_table[k].items():
                    print(f"{i:<4}: {v:+.4f}", end='    ')

                print()

            print(f"\nToday Date: {self.today_date}\n"
                  f"    Next Action:\n"
                  f"        portfolio holding —> {self.holding_action}\n"
                  f"        portfolio empty   —> {self.empty_action}")
        
            sys.stdout = original_stdout


if __name__ == "__main__":
    from preprocess import prerpocess
    import yfinance as yf

    stock_data = yf.download("0050.TW", period="max", auto_adjust=True)

    stock_data = prerpocess("0050.TW", stock_data, [5, 20])

    q_epsilon_agent = RLAgent(
        policy='q_learning',
        action_policy='epsilon_greedy',
        alpha=0.1, gamma=0.95,
    )

    # q_epsilon_agent.train(stock_data)
    q_epsilon_agent.load_q_table(load_best=True)
    q_epsilon_agent.evaluate_learning(stock_data)
    q_epsilon_agent.write_document()
    # print(q_epsilon_agent.Q_table)
