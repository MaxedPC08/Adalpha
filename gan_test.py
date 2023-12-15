"""
This is the shell code for a reinforcement learning agent using
the gym environment.

It uses the CartPole simulation as the example.
true1 is best

author: Max Clemetsen
date: Fall 2023
version: python 3.10.13, gymnasium 0.28.1
"""

import random as r
from collections import deque
import numpy as np
import gymnasium as gym
import tensorflow as tf
from utils import *


ENV_NAME = "CartPole-v1"

"""
General information about CartPole simulation info

"The STATE/OBSERVATION Space"
i	Observation               Min       Max
-   --------------------   --------   -------
0	Cart Position             -2.4       2.4
1	Cart Velocity             -Inf       Inf
2	Pole Angle             ~ -41.8°   ~ 41.8°
3	Pole Velocity At Tip      -Inf       Inf

"Action Space"
0: push to left
1: push to right
"""

# CONSTANTS will go here
# (these are some examples that you might find helpful to use)
# (these values are "fake")
MEMORY_SIZE = 2000
EXPLORATION_DECAY = 0.8
LEARNING_SIZE = 500
LEARNING_RATE = 0.2
GAMMA = 0.9
BATCH_SIZE = 128
EPOCHS = 10
NUMBER_OF_TRAINING_CYCLES = 0
NUM_TESTS = 10
LEARNING_LIMIT = 75  # less than probability (1-100) of performing learning
MAX_STEPS_FOR_SAVING = 5
AUTO_SAVE = True
EXPLORATION_RATE = 0.7

class MaxExpLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.exp_weights = self.add_weight(shape=(input_shape[-1],), initializer=tf.keras.initializers.Constant(1), name="max_exp_weight")

    def call(self, x):
        return tf.sign(x)*(tf.abs(x)**self.exp_weights)


class RLAgent:
    """
    This agent contains the NN model as well as associated methods to help it learn.
    """

    def __init__(self, obs_space:int, acts:int, callback:list, optimizer:tf.keras.optimizers.Optimizer):
        """
        This method is called when the class is constructed. It sets the simulation
        parameters, resets the training data, and loads (or creates) and compiles the NN model.
        """
        self.reset_training_data()
        self.observation_space = obs_space
        self.action_space = acts
        self.callback = callback
        self.optimizer = optimizer


        # load the NN model
        file_name = input("Enter name of model to load (leave blank to start fresh): ")
        if file_name != "":
            self.model = tf.keras.models.load_model(file_name, custom_objects={"max_exp_layer": MaxExpLayer}) # , custom_objects={"MaxExpLayer": MaxExpLayer}
        else:
            in_layer = tf.keras.layers.Input(shape=(self.observation_space,))
            model = MaxExpLayer()(in_layer)
            model = tf.keras.layers.Dense(25,
                        activation=tf.nn.tanh,
                        use_bias=True)(model)
            model = tf.keras.layers.Dense(self.action_space, activation="linear", use_bias=False)(model)
            self.model = tf.keras.Model(in_layer, model)

        self.model.compile(
            loss="mean_squared_error",
            optimizer=self.optimizer
        )
        self.model.summary()

    def reset_training_data(self):
        """
        Resets the training data (self.memory and self.exploration_rate).
        """
        self.exploration_rate = EXPLORATION_RATE
        self.memory = deque(maxlen=MEMORY_SIZE)

    def store_sim_results(self, s, r, a, ns):
        """
        Stores results.
        s - previous state
        r - reward
        a - action
        ns - next state
        """
        self.memory.append((s, r, a, ns))

    def learn(self):
        """Train the NN"""

        # don't do anything until you have enough data
        if len(self.memory)<LEARNING_SIZE:
            print("not enough data "+str(len(self.memory)))
            return
        # pick random data from all saved data to use to improve the model
        # batch is the size of how much data you will use to train the model
        batch = r.sample(self.memory, LEARNING_SIZE)
        for i in range(EPOCHS):
            #Homogenize Data
            states_batch = np.asarray([i[0][0] for i in batch])
            next_state = np.asarray([i[3][0] for i in batch])
            rewards = np.asarray([i[1] for i in batch])
            actions = np.asarray([i[2] for i in batch])
            nn_outs = self.model.predict(states_batch, verbose=0)
            #Calculate Q values
            q_vals = nn_outs[np.arange(len(nn_outs)), actions] * (1-LEARNING_RATE) + LEARNING_RATE * (rewards + GAMMA * np.max(self.model(next_state), axis=1))
            nn_outs[np.arange(len(nn_outs)), actions] = q_vals


            # use each set of data to improve the Q values (training y values)
            # this is where you need to implement the q-learning algorithm
            # Q_new = (1-alpha)*Q_old + alpha*(reward + gamma*maxQ(future))



            # update the NN model
            # note that you will fit the "batch" all at once
            # note that "batch" is not the same as the fitting "batch_size" inside tensorflow...
            self.model.fit(
                states_batch,
                nn_outs,
                epochs=1,
                verbose=0,
                callbacks=self.callback
            )

        # update the exploration value
        self.exploration_rate *= EXPLORATION_DECAY

    def save(self, name="model.h5"):
        """save the NN"""
        if AUTO_SAVE:
            self.model.save(name)
        else:
            file_name = input(
                "Enter model file name with a .h5 extension. Enter nothing to skip saving.:"
            )
            if file_name != "":
                self.model.save(file_name)

    def get_nn_action(self, single_state):
        """use the NN to get an action"""
        raw_predict = self.model.predict(single_state, verbose=0)[0]
        current_action = np.argmax(raw_predict)
        return current_action

    def get_action(self, single_state):
        """This method will determine the next action while training"""

        # exploring
        if np.random.rand() <= self.exploration_rate:
            an_action = r.choice([0, 1])
            # you can chose to use a different algorithm to better train your model
            #
            #
            #
            #
            # ===== YOU NEED TO WRITE THIS CODE =====
            #
            #
            #
            #
            #
            return an_action

        # exploiting
        return self.get_nn_action(single_state)

def calc_reward(state: np.ndarray, reward: float, sim_step: int, next_state: np.ndarray) -> float:
    """
    Calculates the reward for a given state, reward, simulation step, and next state.

    Parameters
    ----------
    state : np.ndarray
        The current state of the simulation.
    reward : float
        The calculated reward.
    sim_step : int
        The current simulation step.
    next_state : np.ndarray
        The next state of the simulation after the action.

    Returns
    -------
    float
        The calculated reward.

    """
    return (3 - np.abs(next_state[0][0]) - next_state[0][2]**2 - np.abs(next_state[0][3])**2) * 0.999**sim_step



def train(callback, optimizer):
    """This is the main program."""
    # setup the simulation
    env = gym.make(
        ENV_NAME)  # include render_mode="human" if you want to see the action
    env._max_episode_steps = 2000  # this is our ultimate goal...
    # get size of observation (states) and action spaces
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    # create my agent
    the_agent = RLAgent(observation_space, action_space, callback, optimizer)
    # reset the simulation
    # state = [position of cart, velocity of cart, angle of pole, Pole Velocity At Tip]
    # to get the real sim_state, we need to only get the first term, and reformat the shape
    # for our NN
    # setup the simulation loop (one loop per step/action)

    # Create some preliminary training data

    for sim_run in range(MEMORY_SIZE//10):
        # reset the simulation
        sim_state = env.reset()
        # state = [position of cart, velocity of cart, angle of pole, Pole Velocity At Tip]
        # to get the real sim_state, we need to only get the first term, and reformat the shape
        # for our NN
        sim_state = np.reshape(sim_state[0], [1, observation_space])
        # setup the simulation loop (one loop per step/action)

        sim_step = 0
        while True:  # keep going until failure
            # increase step
            sim_step += 1

            # Get next action given sim state
            action = the_agent.get_action(sim_state)

            # Calculate next sim states and update to the_agent's memory
            sim_state_next, reward, sim_done, truncated, info = env.step(action)
            sim_state_next = np.reshape(sim_state_next, [1, observation_space])

            the_agent.store_sim_results(sim_state, calc_reward(sim_state, reward, sim_step, sim_state_next), action,
                                        sim_state_next)

            if sim_done:
                break


    # this is the loop of training cycles/runs
    for sim_run in range(NUMBER_OF_TRAINING_CYCLES):
        # reset the simulation
        sim_state = env.reset()
        # state = [position of cart, velocity of cart, angle of pole, Pole Velocity At Tip]
        # to get the real sim_state, we need to only get the first term, and reformat the shape
        # for our NN
        sim_state = np.reshape(sim_state[0], [1, observation_space])
        # setup the simulation loop (one loop per step/action)

        sim_step = 0
        while True:  # keep going until failure
            sim_step += 1
            #env.render()  # comment this line to hide visualizations of simulations
            #               (not recommended to render during training since rendering
            #                can take a long time - but may want to see the simulation
            #                running when first coding)

            # get the desired action based on our model and current state
            action = the_agent.get_action(sim_state)

            # do the action and get the results
            # the environment returns the following tuple:
            #   o the next state of the system after the action
            #   o the reward (always == 1 -- so not really useful?)
            #   o a boolean to indicate if the simulation ended (pole tipped too far
            #     or reached max steps)
            #   o some info value we never use
            sim_state_next, reward, sim_done, truncated, info = env.step(action)
            sim_state_next = np.reshape(sim_state_next, [1, observation_space])
            the_agent.store_sim_results(sim_state, calc_reward(sim_state, reward, sim_step, sim_state_next), action, sim_state_next)
            print(sim_step)
            if sim_step>10:
                sim_done = True

            if sim_done:
                # print at the end of the run
                print(f"Run:{sim_run:4},  score: {sim_step:5}")
                # give opportunity to save NN if a very successful run.
                if sim_step > MAX_STEPS_FOR_SAVING:
                    the_agent.save(f"model_{sim_step}_{sim_run}.h5")

                # end the step loop to start a new run
                break


            # copy the state before the next simulation step
            sim_state = sim_state_next.copy()
            # determine if time to "learn"
            # could move this to only learn at the end of a simulation
            if r.randint(0, 100) < LEARNING_LIMIT:
                the_agent.learn()

    # done with all the training
    print("GYM CartPole Training successfully completed.")

    # save the trained NN
    the_agent.save()

    env = gym.make(
        ENV_NAME,
        render_mode="human")

    # Now use our model to show how well it does (or doesn't)
    for trial_num in range(NUM_TESTS):
        sim_state = env.reset()
        env._max_episode_steps = 2000
        sim_state = np.reshape(sim_state[0], [1, observation_space])
        sim_done = False
        sim_step = 0
        while not sim_done:
            env.render()
            sim_step += 1
            sim_state = np.reshape(sim_state, [1, observation_space])
            action = the_agent.get_nn_action(sim_state)  # only get a NN action
            sim_state, a_reward, sim_done, truncated, info = env.step(action)
            if sim_step>2000:
                sim_done = True
            if sim_done:
                print(".")
                print("Trial:", trial_num, ".   Done after ", sim_step, " steps.")
    env.close()

def test ():
    train()

if __name__ == "__main__":
    main()
