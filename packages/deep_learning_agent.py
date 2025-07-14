import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools
from tensorflow import keras
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Input, Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

from typing import Optional

from tqdm import tqdm

class ReplayBuffer:
    def __init__(self, input_size, max_size=int(3e3), *, batch_size=128):
        self.max_size = max_size
        self.batch_size = batch_size   
        
        self.states = np.empty((self.max_size, input_size), dtype=np.int8)
        self.actions = np.empty((self.max_size,), dtype=np.int8)
        self.rewards = np.empty((self.max_size,), dtype=np.float32)
        self.next_states = np.empty((self.max_size, input_size), dtype=np.int8)
        self.next_actions = np.empty((self.max_size,), dtype=np.int8)

        self.dones = np.empty((self.max_size,), dtype=np.int8)

        self.index = 0
        self.full = False

    def store(self, state, action, reward,
              next_state, next_action=None, done: int=0):
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.next_actions[self.index] = next_action
        self.dones[self.index] = done

        self.index = (self.index + 1) % self.max_size
        self.full = True if self.index == 0 else False

    def get_samples(self) -> tuple[np.ndarray, ...]:
        max_idx = self.max_size if self.full else self.index
        batch_idx = np.random.choice(max_idx, self.batch_size, replace=False)

        return (
            self.states[batch_idx],
            self.actions[batch_idx],
            self.rewards[batch_idx],
            self.next_states[batch_idx],
            self.next_actions[batch_idx],
            self.dones[batch_idx],
        )

class ActionPolicy:
    def __init__(self):
        self.ACTIONS = ['buy', 'sell', 'hold']
        
    def choose_action(self, q_values, evaluate=False):
        raise NotImplementedError

class EpsilonGreedy(ActionPolicy):
    def __init__(self, epsilon=1, eps_dec=int(1e-3), eps_min=0.1):
        super().__init__()
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min

    def choose_action(self, q_values, evaluate=False):
        epsilon = self.epsilon if not evaluate else 0

        if np.random.rand() < epsilon: # exploration
            return np.random.randint(len(self.ACTIONS))
        else : # exploitation
            return np.argmax(q_values)

    def decay(self):
        ## version 1: fast drop at the beginning
        # self.epsilon = self.epsilon * (1 - self.eps_dec) \
        #     if self.epsilon > self.eps_min \
        #     else self.eps_min
        ## version 2: gradually drop
        self.epsilon = self.epsilon - self.eps_dec \
                        if self.epsilon > self.eps_min \
                        else self.eps_min 

class SoftmaxMethod(ActionPolicy):
    def __init__(self, tau=1, tau_dec=int(1e-3), tau_min=0.3):
        super().__init__()
        self.tau = tau
        self.tau_dec = tau_dec
        self.tau_min = tau_min

    def choose_action(self, q_values, evaluate=False):
        tau = self.tau if not evaluate else 0.01

        q_values_div_tau = q_values / tau
        # for stability
        q_values_div_tau = q_values_div_tau - np.max(q_values_div_tau)
        exp_q_values = np.exp(q_values_div_tau)
        prob_q_values = exp_q_values / np.sum(exp_q_values)

        return np.random.choice(prob_q_values.shape[0], p=prob_q_values)

    def decay(self):
        ## version 1: fast drop at the beginning
        # self.tau = self.tau * (1 - self.tau_dec) \
        #     if self.tau > self.tau_min \
        #     else self.tau_min
        ## version 2: gradually drop
        self.tau = self.tau - self.tau_dec \
                    if self.tau > self.tau_min \
                    else self.tau_min 

class DRLAgent:
    def __init__(
        self,
        stock_no: str="0050.TW",
        len_avg_days: int=2,
        action_policy: EpsilonGreedy | SoftmaxMethod =None,
        alpha: float=0.001,
        gamma: float=0.9,
        episodes: int=1000,
        batch_size: int=128,
        hasSecondLayer: bool=False,
        replay_freq : int=128,
        sync_freq: int=500,
    ):
        self.stock_no = stock_no

        self.TRENDS = ['up', 'down', 'stable']
        self.VOLUME_STATUS = ['high', 'low', 'normal']
        self.PORTFOLIO_STATUS = ['empty', 'holding']
        self.STATES = list(itertools.product(*[self.TRENDS] * len_avg_days,
                                             self.VOLUME_STATUS,
                                             self.PORTFOLIO_STATUS))

        self.ACTIONS = ['buy', 'sell', 'hold']

        self.action_policy = action_policy
        self.state_size = len(self.TRENDS) * len_avg_days + len(self.VOLUME_STATUS) + 1
        self.action_size = len(self.ACTIONS)

        self.alpha = alpha
        self.gamma = gamma
        self.episodes = episodes
        self.batch_size = batch_size

        self.nL1 = self.action_size
        self.nL2 = None if not hasSecondLayer else self.action_size

        self.replay_freq = replay_freq
        self.sync_freq = sync_freq

        self.weight_dir = os.path.join(os.getcwd(), 'model_weights')
        os.makedirs(self.weight_dir, exist_ok=True)

        self.policy = None
        self.apn = None

    def initialize(self):
        if self.state_size:
            ## Initialize Networks
            self.q_network = self._build_model(self.nL1, self.nL2)
            self.t_network = self._build_model(self.nL1, self.nL2)
            self._sync_qt_networks()

            ## Experience Replay Buffer
            self.memory = ReplayBuffer(input_size=self.state_size,
                                       batch_size=self.batch_size)
        else :
            raise ValueError('Agent has not received state_size yet')
        
    def load_weight(self, load_file):
        if os.path.isfile(load_file):
            self.q_network.load_weights(load_file)
        else :
            raise ValueError(f'{load_file} does not exist')

    def _sync_qt_networks(self):
        self.t_network.set_weights(self.q_network.get_weights())

    def _build_model(self, n1, n2=None):
        layers = [
            Input(shape=(self.state_size,)),
            Dense(n1, activation='relu'),
        ]
        if n2 != None:
            layers.append(Dense(n2, activation='relu'))

        layers.append(Dense(self.action_size, activation='linear'),)

        model = Sequential(layers)
        model.compile(
            optimizer=Adam(learning_rate=self.alpha),
            loss='mse',
        )
        return model

    def _get_reward(self, price, previous_price, buy_price, pre_portfolio, action):
        r = (price - previous_price) / previous_price
        if pre_portfolio == 0:
            if action == 0:
                r = 0

            elif action == 1:
                r = -1

            else : # action == 'hold'
                r = -r
                
        else : # pre_portfolio == 'holding'
            if action == 0:
                r = -1

            elif action == 1:
                if buy_price > 0:
                    r += (price - buy_price) / buy_price
                else :
                    r = 0

                r = r * 2 if r > 0 else r

            else : # action == 'hold'
                if buy_price > 0:
                    r += (price - buy_price) / buy_price
                else :
                    r = 0
                    
        return r

    def _replay_experience(self):
        if self.memory.index < self.batch_size and not self.memory.full:
            return # memory not enough for a batch

        states, actions, rewards, \
        next_states, next_actions, dones = self.memory.get_samples()

        # predict the max next q value
        q_values = self.q_network.predict_on_batch(states)
        next_q_values = self.t_network.predict_on_batch(next_states)
        
        # max next q values: get max action values for batch update
        max_next_q_values = self._get_max_next_q_values(next_q_values, next_actions)

        ## update q values (in batch size)
        q_values[np.arange(self.batch_size), actions] = \
            rewards \
            + (1 - dones) * self.gamma * max_next_q_values

        ## train q-network
        self.q_network.train_on_batch(states, q_values)

    def _get_max_next_q_values(self, next_q_values, next_actions):
        raise NotImplementedError

    def train(self, df):
        print(f"{self.policy} {self.apn} agent:")
        open_prices = df['Open'].values
        close_prices = df['Close'].values
        feature_cols = df.values[:, 2:]

        pbar = tqdm(range(self.episodes), ncols=100)
        for episode in pbar:
            pbar.set_description(f"Episode: {episode + 1}")

            portfolio = 0 # 0 as empty, 1 as holding
            buy_price = 0

            ## Initial state & action
            state = np.append(feature_cols[0], portfolio) # shape=(10,)
            q_values = self.q_network.predict_on_batch(state[np.newaxis, :]) # shape=(1,3)
            action = self.action_policy.choose_action(q_values[0]) # output => int

            for i in range(1, len(df)):
                previous_price = close_prices[i - 1]
                price = open_prices[i]

                ## get reward
                reward = self._get_reward(price, previous_price,
                                          buy_price, portfolio, action)

                ## execute action
                if action == 0 and portfolio == 0: # buy when empty
                    buy_price = price
                    portfolio = 1
                elif action == 1 and portfolio == 1: # sell when holding
                    buy_price = 0
                    portfolio = 0

                done = 0 if (i != len(df) - 1) else 1

                ## set next state & action
                next_state = np.append(feature_cols[i], portfolio)
                next_action = self._get_policy_next_action(next_state[np.newaxis, :])

                ## store experience
                self.memory.store(state, action, reward,
                                  next_state, next_action)

                ## update q value                
                if i % self.replay_freq == 0:
                    self._replay_experience()

                ## update network
                if i % self.sync_freq == 0:
                    self._sync_qt_networks()

                ## update state & action
                state = next_state
                action = self._update_action(next_state[np.newaxis, :] , next_action)

            ## decay
            self.action_policy.decay()

        self.q_network.save_weights(
            os.path.join(self.weight_dir, f'{self.policy}_{self.apn}.weights.h5')
        )
    
    def _get_policy_next_action(self, next_state):
        raise NotImplementedError

    def _update_action(self, next_state, next_action) -> int:
        if next_action == -1: # q learning 
            q_values = self.q_network.predict_on_batch(next_state)
            return self.action_policy.choose_action(q_values[0])
        else :
            return next_action
    
    def evaluate_learning(self, df: pd.DataFrame, initial_cash=10000):
        ## Initial status
        # traditional strategy
        cash_tra = initial_cash
        shares_tra = 0
        values_tra = []

        # agent
        cash = initial_cash
        shares = 0
        values_learning = []
        portfolio = 0

        open_prices = df.iloc[:, 0].to_numpy()
        close_prices = df.iloc[:, 1].to_numpy()
        feature_cols = df.iloc[:, 2:].to_numpy()

        # traditional strategy holds from the beginning
        shares_tra = cash_tra // open_prices[0]
        cash_tra -= shares_tra * open_prices[0]
        values_tra.append(cash_tra + shares_tra * close_prices[0])

        # agent check day0 status to choose actions
        state = np.append(feature_cols[0], portfolio)
        q_values = self.q_network.predict_on_batch(state.reshape(1, -1))
        action = self.action_policy.choose_action(q_values[0], evaluate=True)
        values_learning.append(initial_cash) # did not do anything at the first day

        for i in range(1, len(df)):
            opening_p = open_prices[i]
            closing_p = close_prices[i]

            # execute action
            if action == 0 and portfolio == 0: # buy when empty
                shares = cash // opening_p
                cash -= shares * opening_p
                portfolio = 1
            elif action == 1 and portfolio == 1: # sell when holding
                cash += shares * opening_p
                shares = 0
                portfolio = 0

            values_tra.append(cash_tra + shares_tra * closing_p)
            values_learning.append(cash + shares * closing_p)

            # set state & action
            state = np.append(feature_cols[i], portfolio)
            q_values = self.q_network.predict_on_batch(state.reshape(1, -1))
            action = self.action_policy.choose_action(q_values[0], evaluate=True)

        ## plot fig
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.subplots_adjust(right=0.55)
        
        ax.plot(values_tra, label='traditional', alpha=0.3, linestyle='--')
        ax.plot(values_learning, label=f"{self.policy} {self.apn}")
        ax.plot([], [], ' ', label=f'alpha={self.alpha}\ngamma={self.gamma}')
        ax.set_xlabel('Holding Days')
        ax.set_ylabel('Portfolio Value')
        ax.set_title(f'{self.stock_no}')

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
        ax.legend()

        ## save fig
        images_dir = os.path.join(os.getcwd(), 'images')
        os.makedirs(images_dir, exist_ok=True)
        fig_fname = f'{self.policy}_{self.apn}.png'
        fig.savefig(os.path.join(images_dir, fig_fname), bbox_inches='tight')

        ## show fig
        plt.show()

        original_stdout = sys.stdout
        os.makedirs('documents', mode=0o755, exist_ok=True)
        with open(f'documents/{self.policy}_{self.apn}.txt', 'w') as f:
            sys.stdout = f
            print(f"Final value holding:\n"
                  f"    cash={cash_tra}\n"
                  f"    shares={shares_tra * close_prices[-1]}\n"
                  f"    Value={values_tra[-1]}")

            print(f"\nFinal value {self.policy}:\n"
                  f"    cash={cash}\n"
                  f"    shares={shares * close_prices[-1]}\n"
                  f"    Value={values_learning[-1]}\n")
            sys.stdout = original_stdout

        return values_learning[-1]

    def show_performance(self):
        test_data = np.array(self.STATES)

        encoded_data = np.column_stack([
            (test_data[:, 0] == 'down').astype(int),
            (test_data[:, 0] == 'stable').astype(int),
            (test_data[:, 0] == 'up').astype(int),
            (test_data[:, 1] == 'down').astype(int),
            (test_data[:, 1] == 'stable').astype(int),
            (test_data[:, 1] == 'up').astype(int),
            (test_data[:, 2] == 'high').astype(int),
            (test_data[:, 2] == 'low').astype(int),
            (test_data[:, 2] == 'normal').astype(int),
            (test_data[:, 3] == 'holding').astype(int),
        ])        
        
        test_q_values = self.q_network.predict_on_batch(encoded_data)

        test_actions = np.argmax(test_q_values, axis=1)
        conditions = [
            test_actions == 0,
            test_actions == 1,
            test_actions == 2,
        ]
        choices = ['buy', 'sell', 'hold']
        test_actions = np.select(conditions, choices, default='hold')

        original_stdout = sys.stdout
        os.makedirs('documents', mode=0o755, exist_ok=True)
        with open(f'documents/{self.policy}_{self.apn}.txt', 'a') as f:
            sys.stdout = f
            print(f'{'State':<42}|{'Best Action':<15}| q values')
            print('-' * 110)
            for i, v in enumerate(test_data):
                print(f"{str(v):<42}|{test_actions[i]:<15}| ", end='')
                for j, n in enumerate(self.ACTIONS):
                    print(f"{n}:{test_q_values[i, j]:>+8.4f}", end='    ')

                print()

            sys.stdout = original_stdout


class DQNAgent(DRLAgent):
    def __init__(
        self,
        stock_no: str="0050.TW",
        len_avg_days: int=2,
        action_policy: EpsilonGreedy | SoftmaxMethod = None,
        apn: str='',
        alpha: float=0.001,
        gamma: float=0.9,
        episodes: int=1000,
        batch_size: int=128,
        hasSecondLayer: bool=False,
        replay_freq: int=128,
        sync_freq: int=500,
    ):
    # def __init__(
        super().__init__(
            stock_no=stock_no,
            len_avg_days=len_avg_days,
            action_policy=action_policy,
            alpha=alpha,
            gamma=gamma,
            episodes=episodes,
            batch_size=batch_size,
            hasSecondLayer=hasSecondLayer,
            replay_freq=replay_freq,
            sync_freq=sync_freq,
        )
        self.policy = 'DQN'
        self.apn = apn

    def _get_max_next_q_values(self, next_q_values, next_actions):
        return np.max(next_q_values, axis=1)

    def _get_policy_next_action(self, next_state):
        return -1

    # def _update_action(self, next_state, next_action):
    #     q_values = self.q_network.predict_on_batch(next_state)
    #     return self.action_policy.choose_action(q_values[0])
    # def _update_action(self, next_state, next_action) -> int:
    #     return next_action


class DsarsaAgent(DRLAgent):
    def __init__(
        self,
        stock_no: str="0050.TW",
        len_avg_days: int=2,
        action_policy: EpsilonGreedy | SoftmaxMethod = None,
        apn: str='',
        alpha: float=0.001,
        gamma: float=0.9,
        episodes: int=1000,
        batch_size: int=128,
        hasSecondLayer: bool=False,
        replay_freq: int=128,
        sync_freq: int=500,

    ):
        super().__init__(
            stock_no=stock_no,
            len_avg_days=len_avg_days,
            action_policy=action_policy,
            alpha=alpha,
            gamma=gamma,
            episodes=episodes,
            batch_size=batch_size,
            hasSecondLayer=hasSecondLayer,
            replay_freq=replay_freq,
            sync_freq=sync_freq,
        )
        self.policy = 'Dsarsa'
        self.apn = apn

    def _get_max_next_q_values(self, next_q_values, next_actions):
        return next_q_values[np.arange(self.batch_size), next_actions]

    def _get_policy_next_action(self, next_state):
        next_q_values = self.q_network.predict_on_batch(next_state)
        return self.action_policy.choose_action(next_q_values[0])

    # def _update_action(self, next_state, next_action) -> int:
    #     return next_action


if __name__ == "__main__":
    from preprocess import prerpocess, deep_agent_preprocess
    import yfinance as yf

    stock_data = yf.download("0050.TW", period="max", auto_adjust=True)

    stock_data = prerpocess("0050.TW", stock_data, [5, 20])
    stock_data = deep_agent_preprocess(stock_data)

    # print(stock_data)
    dqn_eps_agent = DQNAgent(
        action_policy=EpsilonGreedy(),
        apn='epsilon_greedy',
        alpha=0.001,
        gamma=0.9,
        episodes=100,
    )
    dqn_eps_agent.initialize()
    # dqn_eps_agent.load_weight('model_weights/DQN_epsilon_greedy.weights.h5')
    dqn_eps_agent.train(stock_data)
    dqn_eps_agent.evaluate_learning(stock_data)
    dqn_eps_agent.show_performance()
