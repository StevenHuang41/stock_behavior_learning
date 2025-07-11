import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools
from tensorflow import keras
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Input, Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

from tqdm import tqdm

from sklearn.preprocessing import OneHotEncoder

class ReplayBuffer:
    def __init__(self, input_size, max_size=5e3, *, batch_size=32):
        self.max_size = int(max_size)
        self.batch_size = batch_size   
        
        state_dtype = np.dtype((np.int8, (input_size,)))
        self.states = np.empty((self.max_size,), dtype=state_dtype)
        self.actions = np.empty((self.max_size,), dtype=np.int8)
        self.rewards = np.empty((self.max_size,), dtype=np.float32)
        self.next_states = np.empty((self.max_size,), dtype=state_dtype)
        self.next_actions = np.empty((self.max_size,), dtype=np.int8)
        self.dones = np.empty((self.max_size,), dtype=np.int8)
        self.index = 0
        self.full = False

    def store(self, state, action, reward,
              next_state, done, next_action=None):
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.dones[self.index] = done
        self.next_actions[self.index] = next_action

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
            self.dones[batch_idx],
            self.next_actions[batch_idx],
        )

class ActionPolicy:
    def __init__(self):
        self.ACTIONS = ['buy', 'sell', 'hold']
        
    def choose_action(self, q_values, evaluate=False):
        raise NotImplementedError

class EpsilonGreedy(ActionPolicy):
    def __init__(self, epsilon=1, eps_dec=1e-3, eps_min=0.1):
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
    def __init__(self, tau=1, tau_dec=1e-3, tau_min=0.3):
        self.tau = tau
        self.tau_dec = tau_dec
        self.tau_min = tau_min

    def choose_action(self, q_values, evaluate=False):
        tau = self.tau if not evaluate else 1e-2

        exp_q_values = np.exp(q_values / tau)
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
    def __init__(self, action_policy: ActionPolicy,
                 stock_no="0050.TW", n_TREND=2,
                 state_size=None, action_size=None,
                 alpha=0.001, gamma=0.9,
                 episodes=1000,
                 batch_size_=32):
        self.stock_no = stock_no

        self.action_policy = action_policy
        self.state_size = state_size
        self.action_size = action_size

        self.alpha = alpha
        self.gamma = gamma
        self.episodes = episodes
        self.batch_size = batch_size_

        self.policy = None
        self.apn = None
        self.nL1 = self.action_size
        self.nL2 = None

        self.replay_freq = 32
        self.sync_freq = 100

        self.TRENDS = ['up', 'down', 'stable']
        self.VOLUME_STATUS = ['high', 'low', 'normal']
        self.PORTFOLIO_STATUS = ['empty', 'holding']
        self.STATES = list(itertools.product(*[self.TRENDS] * n_TREND,
                                             self.VOLUME_STATUS,
                                             self.PORTFOLIO_STATUS))

        self.ACTIONS = ['buy', 'sell', 'hold']

        self.weight_dir = os.path.join(os.getcwd(), 'model_weights')
        os.makedirs(self.weight_dir, exist_ok=True)

    def initialize(self):
        if self.state_size:
            ## Initialize Networks
            self.q_network = self._build_mode(self.nL1, self.nL2)
            self.t_network = self._build_mode(self.nL1, self.nL2)
            self._sync_qt_networks()

            ## Experience Replay Buffer
            self.memory = ReplayBuffer(input_size=self.state_size, batch_size=self.batch_size)
        else :
            raise ValueError('Agent has not received state_size yet')
        
    def load_weight(self, load_file):
        if os.path.isfile(load_file):
            self.q_network.load_weights(load_file)
        else :
            raise ValueError(f'{load_file} does not exist')

    def _sync_qt_networks(self):
        self.t_network.set_weights(self.q_network.get_weights())

    def _build_mode(self, n1, n2=None):
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
            # jit_compile=False,
        )
        return model

    def _get_reward(self, price, previous_price, buy_price, portfolio, action):
        ## version 2
        # invalid action
        if action == 0 and portfolio == 1:
            return -2 # buy when holding
        if action == 1 and portfolio == 0:
            return -2 # sell when empty

        # valid action
        if action == 0: # buy
            r = 0.03
            return r

        elif action == 1: # sell
            r = (price - buy_price) / buy_price
            if portfolio == 1: # holding
                r += (price - previous_price) / previous_price

            return r

        else : # hold
            r = (price - previous_price) / previous_price
            if portfolio == 1: # holding
                r += (price - buy_price) / buy_price
            else : # empty
                r *= -1 

            if r > 0: # rewared holding positive behavior
                r *= 1.2 

            return r

    def _replay_experience(self):
        if self.memory.index < self.batch_size and not self.memory.full:
            return # memory not enough for a batch

        states, actions, rewards, \
        next_states, dones, next_actions = self.memory.get_samples()

        # predict the max next q value
        q_values = self.q_network.predict_on_batch(states)
        next_q_values = self.t_network.predict_on_batch(next_states)
        
        # max next q values: max action values for update
        max_next_q_values = self._get_max_next_q_values(next_q_values, next_actions)

        ## update q values
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

            # if episode % 100 == 0:
            #     print('    ', end='')
            #     if self.apn == 'epsilon_greedy':
            #         print(f"episode {episode:>3} progressing, "
            #               f"epsilon {self.action_policy.epsilon:>.4f} ...")
            #     else :
            #         print(f"episode {episode:>3} progressing, "
            #               f"tau {self.action_policy.tau:>.4f} ...")

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

                done = 0 if (i != len(df) - 2) else 1

                ## set next state & action
                next_state = np.append(feature_cols[i], portfolio)
                next_action = self._get_policy_next_action(next_state[np.newaxis, :])

                ## store experience
                self.memory.store(state, action, reward,
                                  next_state, done, next_action)

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

        # print("=" * 10, "Finishing Training", "=" * 10)
        ## save weights
        self.q_network.save_weights(
            os.path.join(self.weight_dir, f'{self.policy}_{self.apn}.weights.h5')
        )
    
    def _get_policy_next_action(self, next_state):
        raise NotImplementedError

    def _update_action(self, next_state, next_action):
        raise NotImplementedError
    
    def evaluate_Dlearning(self, df: pd.DataFrame, initial_cash=10000):
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
        state = (*feature_cols[0], portfolio)
        action = self.choose_action(self.action_policy, state, evaluate=True)
        values_learning.append(initial_cash) # did not do anything at the first day

        while index < len(df):
            price = close_prices[index]

            # set state & action
            state = np.append(feature_cols[index], portfolio)
            q_values = self.q_network.predict_on_batch(state.reshape(1, -1))
            action = self.action_policy.choose_action(q_values[0], evaluate=True)

            # execute action
            if action == 0 and portfolio == 0:
                # buy when empty
                shares = cash // price
                cash -= shares * price
                portfolio = 1
            elif action == 1 and portfolio == 1:
                # sell when holding
                cash += shares * price
                shares = 0
                portfolio = 0

            values_tra.append(cash_tra + shares_tra * df.values[index, 0])
            values_learning.append(cash + shares * df.values[index, 0])

            index += 1

        ## plot fig
        apc = 100 * (values_learning[-1] - initial_cash) / initial_cash
        hpc = 100 * (values_tra[-1] - initial_cash) / initial_cash
        cpc = 100 * (values_learning[-1] - values_tra[-1]) / values_tra[-1]

        apc_c = 'green' if apc >= 0 else 'red'
        hpc_c = 'green' if hpc >= 0 else 'red'
        cpc_c = 'green' if cpc >= 0 else 'red'

        text_diff = 3000 if cpc >= 0 else -3000

        plt.plot(values_tra, label="holding", alpha=0.3, linestyle='--')
        plt.plot(values_learning, label=f"{self.policy} {self.apn}")
        plt.annotate(f"Agent: {values_learning[-1]:.0f}",
                     xy=(len(values_learning), values_learning[-1]),
                     xytext=(len(values_learning) + 500, values_learning[-1] + text_diff),
                     fontsize=12,
                     ha='left',
                     va='center',
                     arrowprops={
                         'arrowstyle': "->",
                         'color': 'orange',
                         'alpha': 0.5,
                     })
        plt.annotate(f"Holding: {values_tra[-1]:.0f}",
                     xy=(len(values_tra), values_tra[-1]),
                     xytext=(len(values_tra) + 500, values_tra[-1] - text_diff),
                     fontsize=12,
                     ha='left',
                     va='center',
                     arrowprops={
                         'arrowstyle': "->",
                         'color': 'blue',
                         'alpha': 0.3,
                     })
        plt.xlabel('Holding Days')
        plt.ylabel('Portfolio value')
        plt.title(f"{self.stock_no}")
        plt.figtext(1.2, 0.8, 'Cash Percentage Change', fontsize=12)
        plt.figtext(1.25, 0.75, f'agent growth:    {apc:^+9.2f} %', fontsize=12, color=apc_c)
        plt.figtext(1.25, 0.7,  f'holding growth:  {hpc:^+9.2f} %', fontsize=12, color=hpc_c)
        plt.figtext(1.25, 0.65, f'relative change: {cpc:^+9.2f} %', fontsize=12, color=cpc_c)
        plt.legend()

        ## save fig
        images_dir = os.path.join(os.getcwd(), 'images')
        os.makedirs(images_dir, exist_ok=True)
        fig_fname = f'{self.policy}_{self.apn}.png'
        plt.savefig(os.path.join(images_dir, fig_fname), bbox_inches='tight')

        ## show fig
        plt.show()

        print(f"Final value holding:\n \
                cash={cash_tra}\n \
                shares={shares_tra * df.values[-1, 0]}\n \
                Value={values_tra[-1]}")
        print(f"Final value {self.policy}:\n \
                cash={cash}\n \
                shares={shares * df.values[-1, 0]}\n \
                Value={values_learning[-1]}")

    def show_performance(self):
        test_q_values = self.q_network.predict_on_batch(encoded_data)

        test_actions = np.argmax(test_q_values, axis=1)
        condiitons = [
            test_actions == 0,
            test_actions == 1,
            test_actions == 2,
        ]
        choices = ['buy', 'sell', 'hold']
        test_actions = np.select(condiitons, choices)
        for i, v in enumerate(test_D_data):
            print(f"{str(v):<29}-> {test_actions[i]}\t", end='')
            for j, n in enumerate(self.ACTIONS):
                if j != len(self.ACTIONS) - 1:
                    print(f"{n}:{test_q_values[i, j]:>+8.4f}", end='\t')
                else :
                    print(f"{n}:{test_q_values[i, j]:>+8.4f}")

        self.evaluate_Dlearning(pre_stock_data)


class DQNAgent(DRLAgent):
    def __init__(self, action_policy,
                 state_size, action_size,
                 alpha=0.001, gamma=0.9,
                 episodes=1000,
                 batch_size_=32, *, apn):
        super().__init__(action_policy,
                         state_size, action_size,
                         alpha, gamma,
                         episodes,
                         batch_size_)
        self.policy = 'DQN'
        self.apn = apn

    # return (32, 1) array
    def _get_max_next_q_values(self, next_q_values, next_actions):
        return np.max(next_q_values, axis=1)

    # return index of action
    def _get_policy_next_action(self, next_state):
        return -1

    # return index of action
    def _update_action(self, next_state, next_action):
        q_values = self.q_network.predict_on_batch(next_state)
        return self.action_policy.choose_action(q_values[0])

class DsarsaAgent(DRLAgent):
    def __init__(self, action_policy,
                 state_size, action_size,
                 alpha=0.001, gamma=0.9,
                 episodes=1000,
                 batch_size_=32, *, apn):
        super().__init__(action_policy,
                         state_size, action_size,
                         alpha, gamma,
                         episodes,
                         batch_size_)
        self.policy = 'Dsarsa'
        self.apn = apn

    # return (32, 1) array
    def _get_max_next_q_values(self, next_q_values, next_actions):
        return next_q_values[np.arange(self.batch_size), next_actions]

    # return index of action
    def _get_policy_next_action(self, next_state):
        next_q_values = self.q_network.predict_on_batch(next_state)
        return self.action_policy.choose_action(next_q_values[0])

    # return index of action
    def _update_action(self, next_state, next_action) -> int:
        return next_action


if __name__ == "__main__":
    from preprocess import prerpocess, deep_agent_preprocess
    import yfinance as yf

    stock_data = yf.download("0050.TW", period="max", auto_adjust=True)

    stock_data = prerpocess(stock_data,
                            hasStockSplited=True,
                            split_date='2025-06-06',
                            split_ratio=4)
    stock_data = deep_agent_preprocess(stock_data)
