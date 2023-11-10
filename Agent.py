import os
import time
import random
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from ModifiedTensorBoard import ModifiedTensorBoard
import tensorflow as tf

# Ensure GPU is being used
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth to True to allocate memory as needed
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Print exception if occurred
        print(e)

class DQNAgent:
    def __init__(self, state_space, action_space, learning_rate, replay_memory_size, model_name, min_replay_memory_size, minibatch_size, discount_factor, update_target_every, saved_model=None):
        self.replay_memory_size = replay_memory_size
        self.model_name = model_name
        self.min_replay_memory_size = min_replay_memory_size
        self.minibatch_size = minibatch_size
        self.discount_factor = discount_factor
        self.update_target_every = update_target_every

        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = learning_rate

        self.model = self.load_or_create_model(saved_model)
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=self.replay_memory_size)
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{self.model_name}-{int(time.time())}")
        self.target_update_counter = 0

    def create_model(self):
        # You can specify the device you want to use explicitly
        with tf.device('/GPU:0'):
            model = Sequential([
                Dense(128, input_shape=(self.state_space,), activation='relu'),
                Dense(128, activation='relu'),
                Dense(128, activation='relu'),
                Dense(self.action_space, activation='linear')
            ])
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def load_or_create_model(self, saved_model):
        if saved_model:
            return keras.models.load_model(saved_model)
        return self.create_model()

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self, terminal_state):
        if len(self.replay_memory) < self.min_replay_memory_size:
            return

        minibatch = random.sample(self.replay_memory, self.minibatch_size)
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states, verbose=0)
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states, verbose=0)

        X = []
        y = []

        for index, (current_state, action, reward, _, done) in enumerate(minibatch):
            new_q = reward + self.discount_factor * np.max(future_qs_list[index]) * (not done)
            current_qs = current_qs_list[index]
            current_qs[action] = new_q
            X.append(current_state)
            y.append(current_qs)

        with tf.device('/GPU:0'):
            self.model.fit(np.array(X), np.array(y), batch_size=self.minibatch_size, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        if terminal_state:
            self.target_update_counter += 1
            if self.target_update_counter >= self.update_target_every:
                self.target_model.set_weights(self.model.get_weights())
                self.target_update_counter = 0

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, self.state_space), verbose=0)



