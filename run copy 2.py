import os
import time
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tqdm import tqdm

from rubiksCube2 import Cube
from ModifiedTensorBoard import ModifiedTensorBoard
from Agent import DQNAgent

# Set seed for reproducibility
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)


MEMORY_FRACTION = 0.40

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = MEMORY_FRACTION
session = tf.compat.v1.Session(config=config)
K.set_session(session)

# Environment settings
EPISODES = 5000
MAX_TURNS = 1#0
TURN_INCREMENTS = 1

# Exploration settings
epsilon = 1  # starting value of epsilon
EPSILON_DECAY = 0.99875
MIN_EPSILON = 0.001

# Training settings
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 1000
MIN_REPLAY_MEMORY_SIZE = 100
MINIBATCH_SIZE = 80
UPDATE_TARGET_EVERY = 10
MODEL_NAME = '2x2_Cube'
MIN_REWARD = -20

# Stats settings
AGGREGATE_STATS_EVERY = 50
SHOW_MODEL_EVERY = 100
SHOW_PREVIEW = True

# Initialize environment and agent
env = Cube()
agent = DQNAgent(env.state_space, env.action_space, 0.001, REPLAY_MEMORY_SIZE, MODEL_NAME, MIN_REPLAY_MEMORY_SIZE, MINIBATCH_SIZE, DISCOUNT, UPDATE_TARGET_EVERY)

# For storing episode rewards
episode_rewards = []

# Training loop
for turn_count in range(TURN_INCREMENTS, MAX_TURNS + 1):
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        
        # Reset environment with a specific number of turns
        current_state, start_shuffle_move = env.shuffle(turn_count)
        episode_reward = 0
        step = 0
        done = False

        step_record = []

        while not done and step < MAX_TURNS * 2:  # The maximum steps are twice the turn count for challenge
            action = np.argmax(agent.get_qs(current_state)) if np.random.random() > epsilon else np.random.randint(0, env.action_space)
            reward, new_state, done = env.step(action)

            # Add the step to the step record
            step_record.append(action)



            episode_reward += reward

            if SHOW_PREVIEW and not episode % SHOW_MODEL_EVERY:
                env.render()

            agent.update_replay_memory((current_state, action, reward, new_state, done))
            agent.train(done)

            if done:
                print("SOVLED THE CUBE!")

            current_state = new_state
            step += 1

        # Append episode reward to the rewards list
        episode_rewards.append(episode_reward)

        # Logging and saving
        if episode % AGGREGATE_STATS_EVERY == 0 or episode == 1:
            average_reward = np.mean(episode_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = np.min(episode_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = np.max(episode_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)
            # Save model only if episode_reward is greater or equal to MIN_REWARD
            if episode_reward >= MIN_REWARD:
                agent.model.save(f'models/{MODEL_NAME}__{max_reward:>7.2f}max__{average_reward:>7.2f}avg__{min_reward:>7.2f}min__{int(time.time())}.model')

        # Save the step record to a file for every episode
        with open(f'step_records/new.txt', 'a') as f:
            f.write(f"{start_shuffle_move}, {step_record}, {done}\n")    

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)
            
        

        K.clear_session()

# No need to manually close the session with modern versions of TensorFlow
