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
import Adalpha
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

class MaxExpLayer(tf.keras.layers.Layer):
    """
    A custom Keras layer for applying element-wise exponentiation to the input.

    Attributes:
        exp_weights: The weights for element-wise exponentiation. Initialized with constant values.

    Methods:
        __init__(self, **kwargs): Initializes the MaxExpLayer instance.
        build(self, input_shape): Builds the layer by creating the exp_weights.
        call(self, x): Applies the element-wise exponentiation to the input x.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        """
        Builds the layer by creating the exp_weights.

        Args:
            input_shape: The shape of the input tensor.

        Returns:
            None
        """

        self.exp_weights = self.add_weight(shape=(input_shape[-1],),
                                           initializer=tf.keras.initializers.Constant(1),
                                           name="max_exp_weight")

    def call(self, x):
        """
        Calculate the element-wise power of the absolute value of input `x` with the exponent weights `self.exp_weights`.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The result of element-wise power of the absolute value of `x` with the exponent weights.
        """
        return tf.sign(x)*(tf.abs(x)**self.exp_weights)


class RLAgent:
    """
    This agent contains the NN model as well as associated methods to help it learn.
    """

    def __init__(self,
                 obs_space:int,
                 acts:int,
                 callback:list,
                 optimizer:tf.keras.optimizers.Optimizer,
                 memory_size:int,
                 exp_decay:float,
                 learning_size:int,
                 learning_rate:float,
                 gamma:float,
                 epochs:int,
                 exploration_rate:float):
        """
        This method is called when the class is constructed. It sets the simulation
        parameters, resets the training data, and loads (or creates) and compiles the NN model.
        """
        self.observation_space = obs_space
        self.action_space = acts
        self.callback = callback
        self.optimizer = optimizer
        self.memory_size = memory_size
        self.exp_decay = exp_decay
        self.learning_size = learning_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epochs = epochs
        self.exploration_rate = exploration_rate
        self.initial_exploration_rate = exploration_rate
        self.reset_training_data()

        # load the NN model

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
        self.exploration_rate = self.initial_exploration_rate
        self.memory = deque(maxlen=self.memory_size)

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
        """
        Trains the model using Q-learning algorithm.

        Returns:
            None.
        """
        # don't do anything until you have enough data
        if len(self.memory)<self.learning_size:
            print("not enough data "+str(len(self.memory)))
            return
        # pick random data from all saved data to use to improve the model
        # batch is the size of how much data you will use to train the model
        batch = r.sample(self.memory, self.learning_size)
        for i in range(self.epochs):
            #Homogenize Data
            states_batch = np.asarray([i[0][0] for i in batch])
            next_state = np.asarray([i[3][0] for i in batch])
            rewards = np.asarray([i[1] for i in batch])
            actions = np.asarray([i[2] for i in batch])
            nn_outs = self.model.predict(states_batch, verbose=0)
            #Calculate Q values
            q_vals = (nn_outs[np.arange(len(nn_outs)), actions] *
                      (1-self.learning_rate) + self.learning_rate *
                      (rewards + self.gamma * np.max(self.model(next_state), axis=1)))
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
                batch_size=self.learning_size,
                epochs=1,
                verbose=0,
                callbacks=self.callback
            )

        # update the exploration value
        self.exploration_rate *= self.exp_decay

    def save(self, name="model.h5"):
        """
        Saves the model to a file with the given name.

        :param name: The name of the file to save the model to. Default is "model.h5".
        :type name: str
        :return: None
        """
        if name != "model.h5":
            self.model.save(name)
        else:
            file_name = input(
                "Enter model file name with a .h5 extension. Enter nothing to skip saving.:"
            )
            if file_name != "":
                self.model.save(file_name)

    def get_nn_action(self, single_state):
        """
        Get the action predicted by the neural network model for a single state.

        Parameters:
            single_state (numpy array): The input state for which the action is to be predicted.

        Returns:
            int: The predicted action for the given state.
        """
        raw_predict = self.model.predict(single_state, verbose=0)[0]
        current_action = np.argmax(raw_predict)
        return current_action

    def get_action(self, single_state):
        """
        Generates the action to be taken based on the given state.

        Args:
            single_state: The state for which the action needs to be determined.

        Returns:
            int: The action to be taken.

        Raises:
            None
        """
        # exploring
        if np.random.rand() <= self.exploration_rate:
            an_action = r.choice([0, 1])
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



def train(callback,
          optimizer,
          memory_size=10000,
          cycles=30,
          tests=10,
          learning_probability=0.7,
          epochs=10,
          learning_size=400,
          learning_rate=0.2,
          gamma=0.9,
          exp_decay=0.995,
          exploration_rate=0.8,
          max_steps=1000):
    """
    Trains a reinforcement learning agent using the given callback function, optimizer, and hyperparameters.

    Parameters:
    - callback: A callback function used for updating the agent's model during training.
    - optimizer: An optimizer object used for updating the agent's model parameters.
    - memory_size: The size of the agent's memory buffer (default: 10000).
    - cycles: The number of training cycles/runs (default: 20).
    - tests: The number of test trials to run after training (default: 10).
    - learning_probability: The probability of performing a learning update during each simulation step (default: 0.7).
    - epochs: The number of epochs to train the agent's model during each learning update (default: 10).
    - learning_limit: The maximum number of recent experiences to use for learning (default: 70).
    - learning_rate: The learning rate used by the optimizer (default: 0.001).
    - gamma: The discount factor for future rewards in the agent's Q-values (default: 0.9).
    - exp_decay: The decay rate for the exploration rate (default: 0.995).
    - exploration_rate: The initial exploration rate for selecting actions (default: 0.5).

    Returns:
    - results: A list of the number of steps taken in each test trial after training.
    """
    # setup the simulation
    env = gym.make(
        ENV_NAME)  # include render_mode="human" if you want to see the action
    env._max_episode_steps = max_steps  # this is our ultimate goal...
    # get size of observation (states) and action spaces
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    # create my agent
    the_agent = RLAgent(observation_space,
                        action_space,
                        callback,
                        optimizer=optimizer,
                        epochs=epochs,
                        learning_size=learning_size,
                        learning_rate=learning_rate,
                        gamma=gamma,
                        memory_size=memory_size,
                        exp_decay=exp_decay,
                        exploration_rate=exploration_rate)
    # reset the simulation
    # state = [position of cart, velocity of cart, angle of pole, Pole Velocity At Tip]
    # to get the real sim_state, we need to only get the first term, and reformat the shape
    # for our NN
    # setup the simulation loop (one loop per step/action)

    # Create some preliminary training data

    for sim_run in range(learning_size*2):
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
    for sim_run in range(cycles):
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
            the_agent.store_sim_results(sim_state,
                                        calc_reward(sim_state, reward, sim_step, sim_state_next), action,
                                        sim_state_next)
            if sim_step>max_steps:
                sim_done = True

            if sim_done:
                # print at the end of the run
                print(f"Run:{sim_run:4},  score: {sim_step:5}")

                # end the step loop to start a new run
                break


            # copy the state before the next simulation step
            sim_state = sim_state_next.copy()
            # determine if time to "learn"
            # could move this to only learn at the end of a simulation
            if r.random() < learning_probability:
                the_agent.learn()

    # done with all the training
    print("GYM CartPole Training successfully completed.")

    # save the trained NN

    env = gym.make(
        ENV_NAME,
        render_mode="human")

    # Now use our model to show how well it does (or doesn't)
    seeds = [123, 23, 34, 45, 56, 67, 78, 89, 90, 100]
    results = []
    for trial_num in range(tests):
        sim_state = env.reset(seed = seeds[trial_num])
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
                results.append(sim_step)
                print("Trial:", trial_num, ".   Done after ", sim_step, " steps.")
    env.close()
    del the_agent
    return results

def cartpole_test(callback,
                  optimizer,
                  epochs=50,
                  learning_rate=0.01,
                  ema_w=0.99,
                  change=0.99,
                  adjustment_exp=2,
                  memory_size=10000,
                  cycles=30,
                  tests=10,
                  learning_probability=0.7,
                  learning_size=400,
                  rl_learning_rate=0.2,
                  gamma=0.9,
                  exp_decay=0.995,
                  exploration_rate=0.8):
    """
    Executes the cartpole test with the given parameters.

    Args:
        callback (object): The callback object for the optimizer.
        optimizer (object): The optimizer object.
        epochs (int, optional): The number of epochs to run the test. Defaults to 50.
        learning_rate (float, optional): The learning rate for the optimizers. Defaults to 0.01.
        ema_w (float, optional): The exponential moving average weight for the callback. Defaults to 0.99.
        change (float, optional): The change threshold for the callback. Defaults to 0.99.
        adjustment_exp (int, optional): The punishment factor for chaos in the optimizer. Defaults to 2.
        memory_size (int, optional): The size of the memory for the optimizer. Defaults to 10000.
        cycles (int, optional): The number of cycles to run the test. Defaults to 30.
        tests (int, optional): The number of tests to run per cycle. Defaults to 10.
        learning_probability (float, optional): The probability of learning during training. Defaults to 0.7.
        learning_size (int, optional): The size of the learning set. Defaults to 400.
        rl_learning_rate (float, optional): The learning rate for the reinforcement learning optimizer. Defaults to 0.2.
        gamma (float, optional): The discount factor for the reinforcement learning optimizer. Defaults to 0.9.
        exp_decay (float, optional): The decay rate for the exploration factor. Defaults to 0.995.
        exploration_rate (float, optional): The exploration rate for the reinforcement learning optimizer. Defaults to 0.8.

    Returns:
        object: The AdAlpha optimizer object.
        object: The Adam optimizer object.
    """
    # Set up your data and model here

    # Create the Adam optimizer
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Create the AdAlpha_Momentum optimizer
    adalpha_optimizer = optimizer(learning_rate=learning_rate, adjustment_exp=adjustment_exp)

    callback = callback(adalpha_optimizer, ema_w=ema_w, change=change)

    adam_results = train(callback=[],
                         optimizer=adam_optimizer,
                         memory_size=memory_size,
                         cycles=cycles,
                         epochs=epochs,
                         tests=tests,
                         learning_probability=learning_probability,
                         learning_size=learning_size,
                         learning_rate=rl_learning_rate,
                         gamma=gamma,
                         exp_decay=exp_decay,
                         exploration_rate=exploration_rate)

    adalpha_results = train(callback=callback,
                            optimizer=adalpha_optimizer,
                            memory_size=memory_size,
                            cycles=cycles,
                            epochs=epochs,
                            tests=tests,
                            learning_probability=learning_probability,
                            learning_size=learning_size,
                            learning_rate=rl_learning_rate,
                            gamma=gamma,
                            exp_decay=exp_decay,
                            exploration_rate=exploration_rate)

    # Plot the results
    plt.plot(adalpha_results, label="AdAlpha")
    plt.plot(adam_results, label="Adam")
    plt.legend()
    plt.show()

    return adalpha_optimizer, adam_optimizer

