import numpy as np

import logging
#logging.basicConfig(level=logging.DEBUG)

import tensorflow as tf

import gym
import gym_mastermind

from tqdm import tqdm

from collections import deque
from collections import Counter

from timeit import default_timer as timer
import time
import random


MIN_REWARD = 0.5  # For model save

# Environment settings
EPISODES = 3000#10_000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes

# For more repetitive results
random.seed(1)
np.random.seed(1)


class MastermindEnv():
    """
    Guess a 4-digits long password where each digit is between 0 and 5.

    After each step the agent is provided with a 4-digits long tuple:
    - '2' indicates that a digit has been correclty guessed at the correct position.
    - '1' indicates that a digit has been correclty guessed but the position is wrong.
    - '0' otherwise.

    The rewards at the end of the episode are:
    0 if the agent's guess is incorrect
    1 if the agent's guess is correct

    The episode terminates after the agent guesses the target or
    12 steps have been taken.
    """

    def __init__(self):
        self.code_values = 6
        self.code_lenght = 2
        self.feedback_values = 3
        self.guess_max = 12

        self.state = np.full([2,self.guess_max,self.code_lenght], -1, dtype="float64")

        self.reward_best = (self.code_lenght*self.feedback_values-1)*3
        self.reward_worst = - (self.code_lenght*self.feedback_values-1)*2
        
        self.target = None
        self.guess_count = None
        self.feedback = None

        self.reset()

    def __get_observation(self, action):
        """
        returns:
            feedback array : lenght = self.code_lenght
            reward int: 1 for color, 2 for color and position
            won bool: is True when action is self.target
        """
        match_idxs = set(idx for idx, ai in enumerate(action) if ai == self.target[idx])
        n_correct = len(match_idxs)
        g_counter = Counter(self.target[idx] for idx in range(self.code_lenght) if idx not in match_idxs)
        a_counter = Counter(action[idx] for idx in range(self.code_lenght) if idx not in match_idxs)
        n_white = sum(min(g_count, a_counter[k])for k, g_count in g_counter.items())

        return (
            np.array([0] * (self.code_lenght - n_correct - n_white) + [1] * n_white + [2] * n_correct),
            1 * n_white + 2 * n_correct,
            len(match_idxs) == self.code_lenght
        )


    def step(self, action):
        self.guess_count += 1
        self.state
        feedback, reward, won = self.__get_observation(action)

        done = won == True or self.guess_count >= self.guess_max

        if won:
            reward = self.reward_best
        elif any(np.array_equal(x, action/(self.code_values-1)) for x in self.state[0]):
            reward = self.reward_worst

        self.state[0,self.guess_count-1] = action/(self.code_values-1)
        self.state[1,self.guess_count-1] = feedback/(self.feedback_values-1)
        return feedback, reward , won, done


    def reset(self):
        self.target, _ = self.random_code()
        self.guess_count = 0
        self.feedback = np.zeros(self.code_lenght, dtype = int)
        return self.feedback


    def random_code(self):
        action_int = np.random.randint(0, self.code_values**self.code_lenght-1)
        return  self.int_to_array_base(action_int), action_int


    def int_to_array_base(self, int_base_10):
        int_base_ = np.base_repr(int_base_10, self.code_values)
        result  = np.array(list(map(int,str(int_base_))))
        return np.pad(result, [self.code_lenght-len(result),0],constant_values=0)


class DQNAgent():
    def __init__(self, env):
        self.variables()

        #main model get trained every step
        self.model = self.create_model(env)

        #the predicting model
        self.target_model = self.create_model(env)
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen = self.replay_mem_size)

        self.target_update_counter = 0


    def create_model(self, env):
        self.model_name = "f_d128_d256_dout"
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=(2,env.guess_max,env.code_lenght)))#(2, 12, 4)#(2,env.guess_max,env.size)
        model.add(tf.keras.layers.Dense(2*env.guess_max*env.code_lenght, activation='relu'))
        model.add(tf.keras.layers.Dense(env.code_values**env.code_lenght*2, activation='relu'))
        model.add(tf.keras.layers.Dense(env.code_values**env.code_lenght, activation='linear'))

        adam = tf.optimizers.Adam(lr=self.learning_rate)

        model.compile(loss="mse", optimizer=adam)#, metrics=['accuracy'])

        return model

    def variables(self):
        self.min_replay_mem_size = 1_000
        self.batch_size = 64
        self.replay_mem_size = 50_000
        self.update_target_every = 5
        self.discount = 0.02
        self.learning_rate = 0.0001

    def save_model(self, file_name = "test_model.h5"):
        self.model.save(file_name)

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(np.expand_dims(state, 0))[0]  #np.array(state).reshape(-1, *state.shape)

    def train(self, terminal_state):
        #create training batch from replay memory
        if len(self.replay_memory) < self.min_replay_mem_size:
            return
        
        batch = random.sample(self.replay_memory, self.batch_size)

        #create new Q lists
        begin_states = np.array([transition[0] for transition in batch])
        qs_list = self.model.predict(begin_states)

        end_states = np.array([transition[3] for transition in batch])
        future_qs_list = self.target_model.predict(end_states)

        next_actions = np.argmax(self.model.predict(end_states), axis=1)

        #create X,Y for model.fit
        X = []
        Y = []

        #fill x and y lists with Q values en states
        for index, (begin_state, action, reward, end_state, done) in enumerate(batch):
            if not done:
                max_future_q = future_qs_list[index, next_actions[index]]#np.max(future_qs_list[index])
                new_q = reward + self.discount * max_future_q
            else:
                new_q = reward
            
            current_qs = qs_list[index]
            current_qs[action] = new_q

            X.append(begin_state)
            Y.append(current_qs)

        #fit model to new Q values
        self.model.fit(np.array(X), np.array(Y),
            batch_size = self.batch_size,
            verbose = 0,
            shuffle = False,
            #callbacks =[self.tensorboard]
            #if terminal_state else None            
        )

        #update 
        if terminal_state:
            self.target_update_counter += 1

        #update predicting model every (self.update_target_every) times
        if self.target_update_counter > self.update_target_every:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


env = MastermindEnv()
agent = DQNAgent(env)

print(agent.model.summary())

episode_rewards = []

for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    #agent.tensorboard.step = episode
    _ = env.reset()

    target = env.target

    episode_reward = 0

    done = False

    while not done:
        env_state_begin = env.state.copy()
        
        current_guess_count = env.guess_count

        if np.random.random() > epsilon:
            action_int = np.argmax(agent.get_qs(env_state_begin))
            action = env.int_to_array_base(action_int)
        else:
            action, action_int = env.random_code()
        
        feedback, reward , won, done = env.step(action)

        agent.update_replay_memory((env_state_begin, action_int, reward, env.state, done))
        agent.train(done)
        
        episode_reward += reward

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
    # Append episode reward to a list and log stats (every given number of episodes)
    episode_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(episode_rewards[-AGGREGATE_STATS_EVERY:])/len(episode_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(episode_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(episode_rewards[-AGGREGATE_STATS_EVERY:])
        #agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        # Save model, but only when min reward is greater or equal a set value
        if average_reward >= MIN_REWARD:
            agent.model.save(f'E:\IOT python env\AI\code workspace\models\{agent.model_name}__{average_reward:_>7.2f}avg_{max_reward:_>7.2f}max_{min_reward:_>7.2f}min__{int(time.time())}.model')
        else:
            agent.model.save(f'E:\IOT python env\AI\code workspace\models\{agent.model_name}__{average_reward:_>7.2f}avg_{max_reward:_>7.2f}max_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)