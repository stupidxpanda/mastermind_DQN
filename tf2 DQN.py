import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
#tf.get_logger().setLevel('ERROR')

from tqdm import tqdm

from collections import deque
from collections import Counter
from collections import namedtuple

import datetime
import time
import random


# Environment settings
EPISODES = 4000#10_000


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
        self.code_lenght = 4
        self.feedback_values = 3
        self.guess_max = 12

        self.reward_best = (self.code_lenght*self.feedback_values-1)*3
        self.reward_worst = - (self.code_lenght*self.feedback_values-1)*2
        
        self.state = None
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
        self.state
        feedback, reward, won = self.__get_observation(action)

        self.state[0,self.guess_count] = action/(self.code_values-1)
        self.state[1,self.guess_count] = feedback/(self.feedback_values-1)

        self.guess_count += 1

        done = won == True or self.guess_count >= self.guess_max

        if won:
            reward = self.reward_best
        #checks for duplicate actions
        elif np.any(np.all(np.isin(self.state[0][:self.guess_count-1],action/(self.code_values-1)), axis = 1)):
            reward = -0

        return feedback, reward , won, done


    def reset(self):
        self.target, _ = self.random_code()
        self.guess_count = 0
        self.state = np.full([2,self.guess_max,self.code_lenght], -1, dtype="float64")
        self.feedback = np.zeros(self.code_lenght, dtype = int)
        return self.feedback


    def random_code(self):
        action_int = np.random.randint(0, self.code_values**self.code_lenght-1)
        return  self.int_to_array_base(action_int), action_int


    def int_to_array_base(self, int_base_10):
        int_base_ = np.base_repr(int_base_10, self.code_values)
        result  = np.array(list(map(int,str(int_base_))))
        return np.pad(result, [self.code_lenght-len(result),0],constant_values=0)


class DoubleDQNAgent():
    def __init__(self, env, logger):
        #set training values
        self.min_replay_mem_size = 1_000
        self.batch_size = 64
        self.replay_mem_size = 50_000
        self.update_target_every = 5
        self.discount = 0.2
        self.learning_rate = 0.0001
        self.epsilon = 1
        self.EPSILON_DECAY = 0.99975
        self.MIN_EPSILON = 0.001

        #main model
        self.model = self.create_model(env)

        #the future q prediction model
        self.target_model = self.create_model(env)
        self.target_model.set_weights(self.model.get_weights())

        #
        self.replay_memory = deque(maxlen = self.replay_mem_size)

        self.target_update_counter = 0
        #self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logger.train_log_dir.replace("/", "\\")+ "\\fit", histogram_freq=50, )



    def create_model(self, env):
        self.model_name = "f_d128_d256_dout"
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=(2,env.guess_max,env.code_lenght)))
        model.add(tf.keras.layers.Dense(2*env.guess_max*env.code_lenght, activation='relu'))
        model.add(tf.keras.layers.Dense(env.code_values**env.code_lenght*2, activation='relu'))
        model.add(tf.keras.layers.Dense(env.code_values**env.code_lenght, activation='linear'))

        adam = tf.optimizers.Adam(lr=self.learning_rate)

        model.compile(loss="mse", optimizer=adam)#, metrics=['accuracy'])

        return model


    def add_to_replay_memory(self, transition):
        self.replay_memory.append(transition)


    def get_qs(self, state):
        """
        call for with only state
        returns a list whit q values for that state
        """
        return self.model.predict(np.expand_dims(state, 0))[0]


    def train(self):
        #create training batch from replay memory
        if len(self.replay_memory) < self.min_replay_mem_size:
            return
        
        batch = random.sample(self.replay_memory, self.batch_size)

        #create new Q lists
        begin_states = np.array([transition[0] for transition in batch])
        qs_list = self.model.predict(begin_states)

        end_states = np.array([transition[3] for transition in batch])
        future_qs_list = self.target_model.predict(end_states)

        #predict which action to take for the end state (used to calulate future q)
        next_actions = np.argmax(self.model.predict(end_states), axis=1)

        #create X,Y for model.fit
        X = []
        Y = []

        #fill x and y lists with Q values en states
        for index, (begin_state, action, reward, end_state, done) in enumerate(batch):
            if not done:
                future_q = future_qs_list[index, next_actions[index]]
                new_q = reward + self.discount * future_q
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
            shuffle = False#,
            #callbacks=[self.tensorboard_callback]
        )


    def end_episode(self):
        # Decay epsilon
        self.epsilon = max(self.MIN_EPSILON, self.epsilon*self.EPSILON_DECAY)

        #update predicting model every (self.update_target_every) times
        self.target_update_counter += 1

        if self.target_update_counter > self.update_target_every:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


class stats_logger():
    def __init__(self, training):
        self.training = training
        self.model = None
        self.AGGREGATE_STATS_EVERY = 50
        self.save_model_path = "E:\IOT python env\AI\code workspace\models"
        self.train_log_dir = 'E:/IOT python env/AI/code workspace/logs/train/' + datetime.datetime.now().strftime("%m%d-%H%M%S")
        self.test_log_dir = 'E:/IOT python env/AI/code workspace/logs/test/' + datetime.datetime.now().strftime("%m%d-%H%M%S")
        self.set_writer(training)



        self.episode_rewards = []
        self.end_quess_counts = []
        self.win_count = []
        self.step_num = 0
        self.return_values = namedtuple("return_values", ["average_reward","min_reward","max_reward","average_quess_count","min_quess_count","max_quess_count","winrate"])


    def set_model(self, model):
        self.model = model


    def set_writer(self, training):
        self.training = training
        if self.training:
            self.summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        else:
            self.summary_writer = tf.summary.create_file_writer(self.test_log_dir)


    def update(self,won, reward, quess_count, episode):
        self.step_num += 1
        self.episode_rewards.append(reward)
        self.end_quess_counts.append(quess_count)
        self.win_count.append(won)
        if not episode % self.AGGREGATE_STATS_EVERY or episode == 1:
            self.write_to_file(self.AGGREGATE_STATS_EVERY)
            if self.training:
                self.save_model(self.model, self.AGGREGATE_STATS_EVERY)


    def __calc_values(self, num_episodes):
        average_reward = sum(self.episode_rewards[-num_episodes:])/len(self.episode_rewards[-num_episodes:])
        min_reward = min(self.episode_rewards[-num_episodes:])
        max_reward = max(self.episode_rewards[-num_episodes:])

        average_quess_count = sum(self.end_quess_counts[-num_episodes:])/len(self.end_quess_counts[-num_episodes:])
        min_quess_count = min(self.end_quess_counts[-num_episodes:])
        max_quess_count = max(self.end_quess_counts[-num_episodes:])

        winrate = sum(self.win_count[-num_episodes:])/len(self.win_count[-num_episodes:])

        return self.return_values(average_reward,min_reward,max_reward,average_quess_count,min_quess_count,max_quess_count,winrate)


    def write_to_file(self, stat_calc_nm):
        with self.summary_writer.as_default():
            values = self.__calc_values(stat_calc_nm)
            tf.summary.scalar('average_reward', values.average_reward, step=self.step_num)
            tf.summary.scalar('min_reward', values.min_reward, step=self.step_num)
            tf.summary.scalar('max_reward', values.max_reward, step=self.step_num)
            tf.summary.scalar('average_quess_count', values.average_quess_count, step=self.step_num)
            tf.summary.scalar('min_quess_count', values.min_quess_count, step=self.step_num)
            tf.summary.scalar('max_quess_count', values.max_quess_count, step=self.step_num)
            tf.summary.scalar('winrate', values.winrate, step=self.step_num)
        

    def save_model(self, model, stat_calc_nm):
        if not model == None:
            values = self.__calc_values(stat_calc_nm)
            data_str = f"\{values.average_reward:_>7.2f}avg_reward__{values.average_quess_count:_>7.2f}avg_quess_cnt__{values.winrate:_>7.2f}win_rate__{int(time.time())}.model"
            model.save(self.save_model_path + data_str)


env = MastermindEnv()
logger = stats_logger(training = True)
agent = DoubleDQNAgent(env, logger)
logger.set_model(agent.model)

print(agent.model.summary())

for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    _ = env.reset()

    target = env.target

    episode_reward = 0

    done = False

    while not done:
        env_state_begin = env.state.copy()
        
        current_guess_count = env.guess_count

        if np.random.random() > agent.epsilon:
            action_int = np.argmax(agent.get_qs(env_state_begin))
            action = env.int_to_array_base(action_int)
        else:
            action, action_int = env.random_code()
        
        feedback, reward , won, done = env.step(action)

        agent.add_to_replay_memory((env_state_begin, action_int, reward, env.state, done))
        agent.train()
        
        episode_reward += reward

    agent.end_episode()

    logger.update(won, episode_reward, env.guess_count, episode)