from memory_profiler import profile
import sys
from ModifiedTensorBoard import ModifiedTensorBoard

import numpy as np

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
# from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from collections import deque
import time
import random
from tensorflow import keras

import os

# from run import MODEL_NAME, REPLAY_MEMORY_SIZE

# Agent class
class DQNAgent:
	def __init__(self, states, actions, lr, REPLAY_MEMORY_SIZE, MODEL_NAME, MIN_REPLAY_MEMORY_SIZE, MINIBATCH_SIZE, DISCOUNT, UPDATE_TARGET_EVERY, savedModel = None):

		self.REPLAY_MEMORY_SIZE = REPLAY_MEMORY_SIZE # How many last steps to keep for model training
		self.MODEL_NAME = MODEL_NAME # Name of the model
		self.MIN_REPLAY_MEMORY_SIZE = MIN_REPLAY_MEMORY_SIZE # Minimum number of steps in a memory to start training
		self.MINIBATCH_SIZE = MINIBATCH_SIZE # How many steps (samples) to use for training
		self.DISCOUNT = DISCOUNT # How much to discount future rewards
		self.UPDATE_TARGET_EVERY = UPDATE_TARGET_EVERY # Terminal states (end of episodes)

		# State space
		self.state_space = states

		# Action space
		self.action_space = actions

		# Learning rate
		self.learning_rate = lr


		# Main model
		if savedModel is not None:
				self.model = keras.models.load_model(savedModel)
		else:
			self.model = self.create_model()

		# Target network
		self.target_model = self.create_model()
		self.target_model.set_weights(self.model.get_weights())

		# An array with last n steps for training
		self.replay_memory = deque(maxlen=self.REPLAY_MEMORY_SIZE)

		# Custom tensorboard object
		self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(self.MODEL_NAME, int(time.time())))

		# Used to count when to update target network with main network's weights
		self.target_update_counter = 0

		# # Create empty txt to save memory usage
		# f = open("memory_usage.txt", "w")
		# f.close()


	def create_model(self):
		# Tracking memory usage
		model = Sequential()
		model.add(Dense(128, input_shape=(self.state_space,), activation='relu'))
		model.add(Dense(56, activation='relu'))
		model.add(Dense(56, activation='relu'))
		model.add(Dense(self.action_space, activation='linear'))

		model.compile(loss="mse", optimizer='adam')

		# print("Memory usage create_model: ", process.memory_info().rss)


		return model


	# Adds step's data to a memory replay array
	# (observation space, action, reward, new observation space, done)
	def update_replay_memory(self, transition):

		# If the memory is full, remove the oldest element
		if len(self.replay_memory) == self.REPLAY_MEMORY_SIZE:
			self.replay_memory.popleft()
		# Tracking memory usage
		self.replay_memory.append(transition)
		# print("Memory usage update_replay_memory: ", process.memory_info().rss)
		# Write memory usage to file


	# Trains main network every step during episode
	@profile
	def train(self, terminal_state, step):
		
		# Start training only if certain number of samples is already saved
		if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
			return



		# ## Capture Time
		# start_time = time.time()
		# Get a minibatch of random samples from memory replay table
		minibatch = random.sample(self.replay_memory, self.MINIBATCH_SIZE)
		# ## Save time
		# with open("time.txt", "a") as f:
		# 	f.write("Mini batch sampling: " + str(time.time() - start_time) + " seconds\n")


		# Get current states from minibatch, then query NN model for Q values
		# ## Capture Time
		# start_time = time.time()
		current_states = np.array([transition[0] for transition in minibatch])
		# ## Save time
		# with open("time.txt", "a") as f:
		# 	f.write("Get current states: " + str(time.time() - start_time) + " seconds\n")
		
		# ## Capture Time
		# start_time = time.time()
		current_qs_list = self.model.predict(current_states)
		# ## Save time
		# with open("time.txt", "a") as f:
		# 	f.write("Get current qs list: " + str(time.time() - start_time) + " seconds\n")
		

		# Get future states from minibatch, then query NN model for Q values
		# When using target network, query it, otherwise main network should be queried
		# ## Capture Time
		# start_time = time.time()
		new_current_states = np.array([transition[3] for transition in minibatch])
		# ## Save time
		# with open("time.txt", "a") as f:
		# 	f.write("Get new current states: " + str(time.time() - start_time) + " seconds\n")
		# ## Capture Time
		# start_time = time.time()
		future_qs_list = self.target_model.predict(new_current_states)
		# ## Save time
		# with open("time.txt", "a") as f:
		# 	f.write("Get future qs list: " + str(time.time() - start_time) + " seconds\n")

		X = []
		y = []

		# Now we need to enumerate our batches
		# ## Capture Time
		# start_time = time.time()
		for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

			# If not a terminal state, get new q from future states, otherwise set it to 0
			# almost like with Q Learning, but we use just part of equation here
			if not done:
				max_future_q = np.max(future_qs_list[index])
				new_q = reward + self.DISCOUNT * max_future_q
			else:
				new_q = reward

			# Update Q value for given state
			current_qs = current_qs_list[index]
			current_qs[action] = new_q

			# And append to our training data
			X.append(current_state)
			y.append(current_qs)
		# ## Save time
		# with open("time.txt", "a") as f:
		# 	f.write("Enumerate batches: " + str(time.time() - start_time) + " seconds\n")



		## Capture Time
		# start_time = time.time()
		# Fit on all samples as one batch, log only on terminal state
		self.model.fit(np.array(X), np.array(y), batch_size=self.MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)
		## Save time
		# with open("time.txt", "a") as f:
		# 	f.write("Fit on all samples: " + str(time.time() - start_time) + " seconds\n")


		# Update target network counter every episode
		if terminal_state:
			self.target_update_counter += 1

		
		# If counter reaches set value, update target network with weights of main network
		if self.target_update_counter > self.UPDATE_TARGET_EVERY:
			## Capture Time
			# start_time = time.time()
			self.target_model.set_weights(self.model.get_weights())
			self.target_update_counter = 0
			## Save time
			# with open("time.txt", "a") as f:
			# 	f.write("Update target network: " + str(time.time() - start_time) + " seconds\n")
		

	# Queries main network for Q values given current observation space (environment state)
	def get_qs(self, state, state_space):
		# Tracking memory usage
		# process = psutil.Process(os.getpid())
		qs = self.model.predict(np.reshape(state, (1, state_space)))
		# print("Memory usage get_qs: ", process.memory_info().rss)
		# Write memory usage to file
		# with open("memory_usage.txt", "a") as f:
		# 	f.write("Memory usage get_qs: " + str(process.memory_info().rss) + "btyes\n")
			
		return qs
		# return self.model.predict(state)
		# return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]
	
	def print_memory(self, epNumber):
		# Print the memory of different variables
		# print("model size", sys.getsizeof(self.model)/1000000, "MB")
		# print("target_model size", sys.getsizeof(self.target_model)/1000000, "MB")
		# print("replay_memory size", sys.getsizeof(self.replay_memory)/1000000, "MB")

		
		with open("memory_usage.txt", "a") as f:
			f.write(str(epNumber) + "," + str(sys.getsizeof(self.replay_memory)/1000000) + ",MB\n")


		

		