import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#os.environ['TF_CONFIG'] = json.dumps({'cluster': {'worker': ['X.X.X.X:2000', 'X.X.X.X:2000']}, 'task': {'type': 'worker', 'index': 0}})

import tensorflow as tf
#tf.get_logger().setLevel('ERROR')

from tqdm import tqdm

from collections import deque
from collections import Counter
from collections import namedtuple

import datetime
import time
import random

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
#tf.debugging.set_log_device_placement(True)



# Environment settings
model_name = "2*6_120-000_#2"
training_EPISODES = 120_000#120_000
eval_every = 5
eval_every_for = 50

evaluation_EPISODES = 2_000


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

        self.reward_best = (self.code_lenght*self.feedback_values-1)*3
        
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
            reward = -1

        return reward , won, done


    def reset(self):
        self.target, _ = self.random_code()
        self.guess_count = 0
        self.state = np.full([2,self.guess_max,self.code_lenght], -1, dtype="float64")
        self.feedback = np.zeros(self.code_lenght, dtype = int)


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
        self.train_every = 10
        self.min_replay_mem_size = 2_000
        self.replay_mem_size = 40_000
        #training parms
        self.update_target_every = self.train_every*25
        self.discount = 0.5
        self.learning_rate = 0.001
        self.epsilon = 1
        self.EPSILON_DECAY = 0.99997
        self.MIN_EPSILON = 0.01
        self.batch_size = 32

        self.training_sample_size = self.batch_size*self.train_every*2
        self.validation_sample_size = self.batch_size*round(self.train_every*0.3)


        #main model
        self.model = self.create_model(env)
        
        #the future q prediction model
        self.target_model = self.create_model(env)
        self.target_model.set_weights(self.model.get_weights())

        #
        self.replay_memory = deque(maxlen = self.replay_mem_size)

        self.target_update_counter = 0
        self.tensorboard_callback = ModifiedTensorBoard(log_dir=logger.train_log_dir+ "fit")#, update_freq = "batch")#, histogram_freq=1, ) #.replace("/", "\\")+ "fit"


    def create_model(self, env):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=(2,env.guess_max,env.code_lenght)))
        model.add(tf.keras.layers.Dense(2*env.guess_max*env.code_lenght, activation='relu'))
        #model.add(tf.keras.layers.Dense(env.code_values**env.code_lenght*2, activation='relu'))
        model.add(tf.keras.layers.Dense(env.code_values**env.code_lenght, activation='linear'))

        adam = tf.optimizers.Adam(lr=self.learning_rate)

        model.compile(loss="mse", optimizer=adam)#, metrics=['accuracy'])

        return model


    def add_to_replay_memory(self, transition):
        self.replay_memory.append(transition)


    def get_qs(self, state):
        """
        call for with only one state
        returns a list white q values for that state
        """
        return np.squeeze(self.model.predict_on_batch(state[None, :]))


    def train(self):
        #create training batch from replay memory
        if len(self.replay_memory) < self.min_replay_mem_size:
            return
        
        samples = random.sample(self.replay_memory, self.training_sample_size + self.validation_sample_size)

        #create new Q lists
        begin_states = np.array([transition[0] for transition in samples])
        qs_list = np.array(self.model.predict_on_batch(begin_states))

        end_states = np.array([transition[3] for transition in samples])
        future_qs_list = np.array(self.target_model.predict_on_batch(end_states))

        #predict which action to take for the end state (used to calulate future q)
        next_actions = np.argmax(self.model.predict(end_states), axis=1)

        #create X,Y for model.fit
        samples_X = []
        samples_Y = []

        #fill x and y lists with Q values en states
        for index, (begin_state, action, reward, end_state, done) in enumerate(samples):
            if not done:
                future_q = future_qs_list[index, next_actions[index]]
                new_q = reward + self.discount * future_q
            else:
                new_q = reward
            
            current_qs = qs_list[index]
            current_qs[action] = new_q

            samples_X.append(begin_state)
            samples_Y.append(current_qs)

        training_X, training_Y = np.array(samples_X[:-self.validation_sample_size]), np.array(samples_Y[:-self.validation_sample_size])
        validation_X, validation_Y = np.array(samples_X[-self.validation_sample_size:]), np.array(samples_Y[-self.validation_sample_size:])

        #print(f"byte size training_X: {training_X.nbytes}, training_Y: {training_Y.nbytes}, validation_X: {validation_X.nbytes}, validation_Y: {validation_Y.nbytes}")

        #fit model to new Q values
        self.model.fit(training_X, training_Y,
            batch_size = self.batch_size,
            verbose = 0,
            shuffle = False,
            validation_data=(validation_X, validation_Y)
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
    def __init__(self, training, model_name = None):
        self.training = training
        self.AGGREGATE_STATS_EVERY = 50
        if model_name is None:
            self.save_model_path = f"/home/martijn/python/models/{datetime.datetime.now().strftime('%m%d-%H_%M_%S')}/" 
            self.train_log_dir = f"/home/martijn/python/logs/train/{datetime.datetime.now().strftime('%m%d-%H_%M_%S')}/"
            self.test_log_dir = f"/home/martijn/python/logs/test/{datetime.datetime.now().strftime('%m%d-%H_%M_%S')}/"
        else: 
            self.save_model_path = f"/home/martijn/python/models/{model_name}/" 
            self.train_log_dir = f"/home/martijn/python/logs/train/{model_name}/"
            self.test_log_dir = f"/home/martijn/python/logs/test/{model_name}/"
        self.set_writer(training)
        try:
            os.mkdir(f"{self.save_model_path}")
        except:
            pass

        self.best_values = None
        self.episode_rewards = []
        self.end_quess_counts = []
        self.win_count = []
        self.step_num = 0
        self.return_values = namedtuple("return_values", ["average_reward","min_reward","max_reward","average_quess_count","min_quess_count","max_quess_count","winrate"])


    def set_writer(self, training):
        self.training = training
        if self.training:
            self.summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        else:
            self.summary_writer = tf.summary.create_file_writer(self.test_log_dir)


    def update(self,won, reward, quess_count, episode, epsilon):
        self.step_num += 1
        self.episode_rewards.append(reward)
        self.end_quess_counts.append(quess_count)
        self.win_count.append(won)
        if not episode % self.AGGREGATE_STATS_EVERY or episode == 1:
            self.write_to_file(self.AGGREGATE_STATS_EVERY, epsilon)


    def __calc_values(self, num_episodes):
        average_reward = sum(self.episode_rewards[-num_episodes:])/len(self.episode_rewards[-num_episodes:])
        min_reward = min(self.episode_rewards[-num_episodes:])
        max_reward = max(self.episode_rewards[-num_episodes:])

        average_quess_count = sum(self.end_quess_counts[-num_episodes:])/len(self.end_quess_counts[-num_episodes:])
        min_quess_count = min(self.end_quess_counts[-num_episodes:])
        max_quess_count = max(self.end_quess_counts[-num_episodes:])

        winrate = sum(self.win_count[-num_episodes:])/len(self.win_count[-num_episodes:])

        return self.return_values(average_reward,min_reward,max_reward,average_quess_count,min_quess_count,max_quess_count,winrate)


    def write_to_file(self, stat_calc_nm, epsilon):
        with self.summary_writer.as_default():
            values = self.__calc_values(stat_calc_nm)
            tf.summary.scalar('average_reward', values.average_reward, step=self.step_num)
            tf.summary.scalar('min_reward', values.min_reward, step=self.step_num)
            tf.summary.scalar('max_reward', values.max_reward, step=self.step_num)
            tf.summary.scalar('average_quess_count', values.average_quess_count, step=self.step_num)
            tf.summary.scalar('min_quess_count', values.min_quess_count, step=self.step_num)
            tf.summary.scalar('max_quess_count', values.max_quess_count, step=self.step_num)
            tf.summary.scalar('winrate', values.winrate, step=self.step_num)
            if self.training:
                tf.summary.scalar('epsilon', epsilon, step=self.step_num)
        

    def save_model(self, model, stat_calc_nm):
        values = self.__calc_values(stat_calc_nm)
        save_model = False

        if self.best_values is None:
            self.best_values = np.array(values)

        bigger_index = np.where(values > self.best_values)
        smaller_index = np.where(values < self.best_values)

        for index in np.where(np.isin(bigger_index, [0, 1, 2, 6]) == True, bigger_index, None)[0]:
            if not index == None:
                self.best_values[index] = values[index]
                save_model = True

        for index in np.where(np.isin(smaller_index, [3, 5]) == True, smaller_index, None)[0]:
            if not index == None:
                self.best_values[index] = values[index]
                save_model = True

        if save_model == True:
            data_str = f"{self.step_num}_{values.average_reward:_>7.2f}avg_reward_{values.average_quess_count:_>7.2f}avg_quess_cnt_{values.winrate:_>7.2f}win_rate.h5"
            model.save(self.save_model_path + data_str)


def game_loop(env, agent, logger, episode, training):

        env.reset()
        episode_reward, done = 0, False

        while not done:
            env_state_begin = env.state.copy()

            if training and np.random.random() > agent.epsilon:
                action, action_int = env.random_code()
            else:
                action_int = np.argmax(agent.get_qs(env_state_begin))
                action = env.int_to_array_base(action_int)
            
            reward , won, done = env.step(action)

            if training:
                agent.add_to_replay_memory((env_state_begin, action_int, reward, env.state, done))
            
            episode_reward += reward

        logger.update(won, episode_reward, env.guess_count, episode, agent.epsilon)

env = MastermindEnv()
tr_logger = stats_logger(training = True, 
    model_name=model_name
)
eval_logger = stats_logger(training = False,
    model_name=model_name
)
agent = DoubleDQNAgent(env, tr_logger)

print(agent.model.summary())

eval_cnt= 0

for episode in tqdm(range(1, training_EPISODES + 1), ascii=True, unit="tr_epidode"):
    game_loop(
        env = env, 
        agent = agent, 
        logger= tr_logger,
        episode = episode,
        training = True
    )
    agent.end_episode()
    if not episode % agent.train_every:
        agent.train()

    if not episode % eval_every:
        eval_logger.step_num = episode
        eval_cnt += 1
        game_loop(
            env = env, 
            agent = agent, 
            logger= eval_logger,
            episode = episode,
            training = False
        )
        if not eval_cnt % eval_every_for//2:
            eval_logger.save_model(agent.model, eval_every_for)

print("model evalution timeeee!!!")

eval_logger.safe_model=False

for episode in tqdm(range(1, evaluation_EPISODES + 1), ascii=True, unit="eval_epidode"):
    game_loop(
        env = env, 
        agent = agent, 
        logger= eval_logger,
        episode = episode,
        training = False
    )

eval_logger.save_model(agent.model, eval_every_for)