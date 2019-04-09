import numpy as np 
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, BatchNormalization
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
from keras.regularizers import Regularizer
import keras.backend as K
import tensorflow as tf
import random
from .buffer import ReplayBuffer

class ActorCriticModels:
    def __init__(self, task, sess):
        self.env = env
        self.sess = sess
        
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low
        
        self.learning_rate = 0.001
        self.episilon = 1.0
        self.epsilon_decay = .995
        self.gamma = .95
        self.tau   = .125
        
        self.buffer_size = 100000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)
        
        # Noise process
        self.exploration_mu = 0
        self.exploration_theta = 0.15
        self.exploration_sigma = 0.2
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)
        
        # Actors
        self.actor_state_input, self.local_actor_model = self.actor_model()
        _, self.target_actor_model = self.actor_model()
        
        # Define loss function using action value (Q value) gradients
        self.actor_critic_grad = tf.placeholder(tf.float32, [None, self.state_size])
        
        actor_model_weights = self.local_actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.local_actor_model.output, actor_model_weights, -self.actor_critic_grad)
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize =  tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)
        
        
        # Critics
        self.critic_state_input, self.critic_action_input, self.local_critic_model = self.critic_model()
        _, _, self.target_critic_model = self.critic_model()
        
        self.critic_grads = tf.gradients(self.local_critic_model.output, self.critic_action_input)
        
         # Initialize for later gradient calculations
        self.sess.run(tf.initialize_all_variables())       
        
    def actor_model(self):
        # Define input layer (states)
        state_input = Input(shape=(self.state_size,), name='states')
        
        # Add hidden layers
        h1 = Dense(units=64, activation='relu')(state_input)
        h2 = Dense(units=128, activation='relu')(h1)
        h3 = Dense(units=64, activation='relu')(h2)
        h4 = BatchNormalization(gamma_regularizer=regularizers.Regularizer())(h3)
        
        # Try different layer sizes, activations, add batch normalization, regularizers, etc.
        output = Dense(self.action_size, activation='relu')(h4)
        
        # Add final output layer with sigmoid activation
        output_actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low,name='actions')(output)
        
        # Create Keras model
        model = Model(input=state_input, output=output)
        adam  = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        
        return state_input, model
    
    def critic_model(self):
        # Define input layers
        state_input = Input(shape=(self.state_size,), name='states')
        
        state_h1 = Dense(64, activation='relu')(state_input)
        state_h2 = Dense(128)(state_h1)
        
        action_input = Input(shape=(self.action_size,), name='actions')
        action_h1 = Dense(128)(action_input)
        
        # Try different layer sizes, activations, add batch normalization, regularizers, etc.
        net_states = layers.BatchNormalization(gamma_regularizer=regularizers.Regularizer())(state_h2)
        net_actions = layers.BatchNormalization(gamma_regularizer=regularizers.Regularizer())(action_h1) 
        
        merged = Add()([net_states, net_actions])
        merged_h1 = Dense(64, activation='relu')(merged)
        output = Dense(1, activation='relu')(merged_h1)
        model  = Model(input=[state_input,action_input], output=output)
        
        adam  = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        
        return state_input, action_input, model
        
    def trais(self):
        
        if len(self.memory) < batch_size:
            return
        
        rewards = []
        experiences = self.memory.sample()
        self._train_critic(samples)
        self._train_actor(samples)
        
    def _train_critic(self, samples):        
        for sample in samples:
            cur_state, action, reward, new_state, done = sample
            if not done:
                target_action = self.target_actor_model.predict(new_state)
                future_reward = self.target_critic_model.predict([new_state, target_action])[0][0]
                reward += self.gamma * future_reward
                
            self.critic_model.fit([cur_state, action], reward, verbose=0)
            
    def _train_actor(self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, _ = sample
            predicted_action = self.local_actor_model.predict(cur_state)
            grads = self.sess.run(self.critic_grads, feed_dict ={self.critic_state_input:cur_state,self.critic_action_input:predicted_action})[0]
            
            self.sess.run(self.optimize, feed_dict={self.actor_state_input: cur_state, self.actor_critic_grad: grads})
    
    def _update_actor_target(self):
        actor_model_weights  = self.actor_model.get_weights()
        actor_target_weights =self.target_critic_model.get_weights()
        
        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i]
        self.target_critic_model.set_weights(actor_target_weights)
        
    def act(self, cur_state):
        action = self.local_actor_model.model.predict(state)
        return action + self.noise.sample()
        
        
        
        
        
        