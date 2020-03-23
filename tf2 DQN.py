import numpy as np

import logging
#logging.basicConfig(level=logging.DEBUG)

import tensorflow as tf

import gym
import gym_mastermind

from tqdm import tqdm

from collections import deque
from timeit import default_timer as timer
import time
import random


MIN_REWARD = 0.5  # For model save

# Environment settings
EPISODES = 10_000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes

# For more repetitive results
random.seed(1)
np.random.seed(1)

# Own Tensorboard class speeds up TensorBoard for DQN
class ModifiedTensorBoard(tf.keras.callbacks.TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self._log_write_dir = self.log_dir
        self.writer = tf.summary.create_file_writer(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


class DQNAgent():
    def __init__(self, env):
        self.variables()

        #main model get trained every step
        self.model = self.create_model(env)

        #the predicting model
        self.target_model = self.create_model(env)
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen = self.replay_mem_size)

        #self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{self.model_name}-{int(time.time())}")

        self.target_update_counter = 0


    def create_model(self, env):
        self.model_name = "f_d128_d256_dout"
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=(2,env.guess_max,env.size)))#(2, 12, 4)#(2,env.guess_max,env.size)
        model.add(tf.keras.layers.Dense(2*env.guess_max*env.size, activation='relu'))
        model.add(tf.keras.layers.Dense(env.values**env.size*2, activation='relu'))
        model.add(tf.keras.layers.Dense(env.values**env.size, activation='linear'))

        #model.compile(optimizer="adam", loss=tf.keras.losses.MeanAbsoluteError()

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
        current_states = np.array([transition[0] for transition in batch])
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in batch])
        future_qs_list = self.target_model.predict(new_current_states)

        #create X,Y for model.fit
        X = []
        Y = []

        #fill x and y lists with Q values en states
        for index, (current_state, action, reward, new_current_state, done) in enumerate(batch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.discount * max_future_q
                # if index == 1:
                #     print(f"new_q: {new_q} = reward: {reward} + discounted future reward: {self.discount * max_future_q}")
            else:
                new_q = reward
            
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
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


def int_to_array_base(int_base_10, base, array_length):
    int_base_ = np.base_repr(int_base_10, base)
    result  = np.array(list(map(int,str(int_base_))))
    return np.pad(result, [array_length-len(result),0],constant_values=0)


def array_to_reward(array):
    array_str = ''.join(map(str, array))
    ocur_1 = array_str.count('1')
    ocur_2 = array_str.count('2')
    return ocur_1 + ocur_2*2


def is_row_in_array(row, array):
    row = np.array(row, dtype="float32")
    array = np.array(array, dtype="float32")
    ans = any(np.array_equal(x, row) for x in array)
    return ans


env = gym.make('Mastermind-v0') 
agent = DQNAgent(env)

print(agent.model.summary())

episode_rewards = []
reward_best = (3**env.size-1)
reward_worst = -(3**env.size-1)

for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    #agent.tensorboard.step = episode
    _ = env.reset()

    env_state_end = np.full([2,env.guess_max,env.size], -1)

    target = env.target

    episode_reward = 0

    done = False

    while not done:
        env_state_begin = env_state_end.copy()
        
        current_guess_count = env.guess_count

        if np.random.random() > epsilon:
            action = np.argmax(agent.get_qs(env_state_begin))
        else:
            action = np.random.randint(0, env.values**env.size-1)

        action_env = int_to_array_base(action, env.values, env.size)
        
        env_state_end[0,current_guess_count] = np.array(action_env)/(env.values-1)
        feedback,won,done,_ = env.step(list(action_env))
        env_state_end[1,current_guess_count] = np.array(feedback)/2

        if won:
            reward = reward_best*2
        else:
            if is_row_in_array(action_env/(env.values-1), env_state_begin[0]):
                reward = reward_worst
            else:
                reward = array_to_reward(feedback)

        agent.update_replay_memory((env_state_begin, action, reward, env_state_end, done))
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