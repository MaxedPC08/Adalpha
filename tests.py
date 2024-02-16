import matplotlib.pyplot as plt

from tests_core import *
from utils import *

FILE_NAME = "SeoulBikeData.csv"
SPLIT = 0.67

def bike_test(callback, optimizer, epochs=100, learning_rate=0.01, ema_w=0.9, change=0.99, adjustment_exp=7):
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Model Fitting Results at lr={learning_rate} on Bike Data")
    adalpha_r_2, adalpha_y_pred, adalpha_y_test = adalpha_train_bike(callback=callback, optimizer=optimizer,
                                                                     epochs=epochs, learning_rate=learning_rate,
                                                                     adjustment_exp=adjustment_exp, ema_w=ema_w,
                                                                     change=change)
    r_2, y_pred, y_test= adam_train_bike(epochs, learning_rate)
    plt.legend()
    plt.show()

    plt.title(f"Predictions vs Actual at lr={learning_rate}\nOver {epochs} epochs")
    plt.xlabel("Training Data")
    plt.ylabel("Model Data")
    plt.plot(y_test, y_pred, "g.", label="Adam Predictions")
    plt.plot(adalpha_y_test, adalpha_y_pred, "r.", label="Adalpha Predictions")
    plt.plot([0, 3500], [0, 3500], "b-", label="Perfect Precictions")
    plt.legend()
    plt.show()
    print(f"Adalpha r squared score: {adalpha_r_2}\nAdam r squared score: {r_2}")
    return adalpha_r_2, r_2

def bike_multiple_test(callback,
                       optimizer,
                       epochs=100,
                       learning_rate=0.01,
                       ema_w=0.9,
                       change=0.99,
                       adjustment_exp=6,
                       tests=10,
                       copy=False):
    """
    Performs multiple tests on the bike data using the given parameters.

    Parameters:
        callback (function): The chaos callback function to use.
        optimizer (function): The chaos optimizer function to use.
        epochs (int): The number of epochs to train for.
        learning_rate (float): The learning rate to use.
        ema_w (float): The EMA weight to use.
        change (float): The change value to use.
        adjustment_exp (int): The chaos punishment value to use.
        tests (int): The number of tests to run.
        copy (bool): Whether to copy the results to the clipboard in Excel format.

    Returns:
        None
    """
    losses = []
    for i in range(tests):
        losses.append(
            bike_test(callback=callback, optimizer=optimizer, epochs=epochs, learning_rate=learning_rate,
                      ema_w=ema_w, change=change, adjustment_exp=adjustment_exp))
    if copy:
        pd.DataFrame(losses).to_clipboard(excel=True)
    return losses

def bike_chaos_test(callback,
                    optimizer,
                    epochs=50,
                    learning_rate=0.01,
                    ema_w=0.9,
                    change=0.99,
                    adjustment_exp=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]):
    """
    Main executable for the program
    :return: None
    """
    adalpha_r_2 = []
    for val in adjustment_exp:
        adalpha_r_2.append(adalpha_train_bike(callback=callback, optimizer=optimizer, epochs=epochs,
                                              learning_rate=learning_rate, adjustment_exp=val, ema_w=ema_w,
                                              change=change)[0])

    plt.clf()
    # Graphing the Adalpha Results
    plt.xlabel("Chaos Punishment")
    plt.ylabel("Loss")
    plt.title(f"R Squared vs Chaos Punishment\nOver {epochs} epochs on SBSDD")
    plt.plot(adjustment_exp, adalpha_r_2, "r-", label="Adalpha R2")
    plt.legend()
    plt.grid(True)
    plt.show()

def bike_ema_w_test(callback,
                    optimizer,
                    epochs=50,
                    learning_rate=0.01,
                    ema_w=[0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99],
                    change=0.99,
                    adjustment_exp=2):
    """
    Main executable for the program
    :return: None
    """
    adalpha_r_2 = []
    for val in ema_w:
        adalpha_r_2.append(adalpha_train_bike(callback=callback, optimizer=optimizer, epochs=epochs,
                                              learning_rate=learning_rate, adjustment_exp=adjustment_exp,
                                              ema_w=val, change=change)[0])

    plt.clf()
    # Graphing the Adalpha Results
    plt.xlabel("ema_w")
    plt.ylabel("Loss")
    plt.title(f"R Squared vs Chaos Punishment\nOver {epochs} epochs")
    plt.plot(ema_w, adalpha_r_2, "r-", label="Adalpha R2")
    plt.legend()
    plt.grid(True)
    plt.show()

def bike_lr_curve(callback,
                  optimizer,
                  epochs=10,
                  learning_rates=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
                  ema_w=0.9,
                  change=0.99,
                  adjustment_exp=6,
                  tests=10,
                  copy=False):
    results = []
    for lr in learning_rates:
        results.append(np.mean(bike_multiple_test(callback=callback, optimizer=optimizer, epochs=epochs,
                                                  learning_rate=lr,
                                                  ema_w=ema_w,
                                                  change=change,
                                                  adjustment_exp=adjustment_exp,
                                                  tests=tests, copy=False), axis=0))

    results = np.asarray(results)
    print(results)
    plt.clf()
    # Graphing the Adalpha Results
    plt.xlabel("lr")
    plt.ylabel("Loss")
    plt.xscale("log")
    plt.yscale("log")
    plt.title(f"Loss vs Learning Rate\nOver {epochs} epochs on Bike Dataset")
    plt.plot(learning_rates, results[:, 0], "r-", label="Adalpha Loss")
    plt.plot(learning_rates, results[:, 1], "b-", label="Adam Loss")
    plt.legend()
    plt.show()
    if copy:
        pd.DataFrame(results).to_clipboard(excel=True)

def mnist_test(callback, optimizer, epochs=5, learning_rate=0.01, ema_w=0.9, change=0.99, adjustment_exp=2):
    """
    Trains a neural network on the MNIST dataset using Adalpha optimization.

    Parameters:
        callback (function): The chaos callback function to use.
        optimizer (function): The chaos optimizer function to use.
        epochs (int): The number of epochs to train for.
        learning_rate (float): The learning rate to use.
        ema_w (float): The EMA weight to use.
        change (float): The change value to use.
        adjustment_exp (int): The chaos punishment value to use.

    Returns:
        A tuple containing the accuracy of the Adalpha model, the predictions from the Adalpha model, and the true labels for the test set.
    """

    adalpha_acc, adalpha_y_pred, adalpha_y_test = adalpha_train_mnist(callback=callback, optimizer=optimizer,
                                                                      epochs=epochs, learning_rate=learning_rate,
                                                                      adjustment_exp=adjustment_exp, ema_w=ema_w,
                                                                      change=change)
    acc, y_pred, y_test = adam_train_mnist(epochs, learning_rate)
    plt.show()
    # ====================
    # USE MODEL TO PREDICT and create a scatterplot of the y and y_pred
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    fig, (ax1, ax2) = plt.subplots(1, 2)

    heatmap_max = make_heatmap(np.argmax(adalpha_y_pred, 1), adalpha_y_test)

    im = ax1.imshow(heatmap_max)

    # Show all ticks and label them with the respective list entries
    ax1.set_xticks(np.arange(len(labels)), labels=labels)
    ax1.set_yticks(np.arange(len(labels)), labels=labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.


    ax1.set_title(f"Predictions vs Actual at\nlr={learning_rate} from Adalpha")

    heatmap = make_heatmap(np.argmax(y_pred, 1), y_test)

    im = ax2.imshow(heatmap)

    # Show all ticks and label them with the respective list entries
    ax2.set_xticks(np.arange(len(labels)), labels=labels)
    ax2.set_yticks(np.arange(len(labels)), labels=labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    ax2.set_title(f"Predictions vs Actual at\nlr={learning_rate} from Adam\nOver {epochs} epochs")
    fig.tight_layout()
    plt.show()
    return adalpha_acc, acc

def mnist_multiple_test(callback,
                        optimizer,
                        epochs=10,
                        learning_rate=0.01,
                        ema_w=0.9,
                        change=0.99,
                        adjustment_exp=4,
                        tests=10,
                        copy=False):
    """
    Runs multiple tests of the MNIST_test function.

    Parameters:
        callback (object): The chaos callback function to use.
        optimizer (object): The chaos optimizer function to use.
        epochs (int): The number of epochs to train for.
        learning_rate (float): The learning rate to use.
        ema_w (float): The EMA weight to use.
        change (float): The change value to use.
        adjustment_exp (int): The chaos punishment value to use.
        tests (int): The number of tests to run.
        copy (bool): Whether to copy the results to the clipboard in Excel format.

    Returns:
        None
    """
    losses = []
    for i in range(tests):
        losses.append(
            mnist_test(callback=callback, optimizer=optimizer, epochs=epochs, learning_rate=learning_rate,
                       adjustment_exp=adjustment_exp, ema_w=ema_w, change=change))

    if copy:
        pd.DataFrame(losses).to_clipboard(excel=True)
    return losses


def mnist_chaos_test(callback,
                     optimizer,
                     epochs=2,
                     learning_rate=0.01,
                     ema_w=0.9,
                     change=0.99,
                     adjustment_exp=[0.2, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]):
    """
    Main executable for the program
    :return: None
    """
    adalpha_r_2 = []
    for val in adjustment_exp:
        adalpha_r_2.append(adalpha_train_mnist(callback=callback, optimizer=optimizer, epochs=epochs,
                                               learning_rate=learning_rate, adjustment_exp=val, ema_w=ema_w,
                                               change=change)[0])

    plt.clf()
    # Graphing the Adalpha Results
    plt.xlabel("Chaos Punishment")
    plt.ylabel("Loss")
    plt.title(f"Accuracy vs Chaos Punishment\nOver {epochs} epochs on MNIST")
    plt.plot(adjustment_exp, adalpha_r_2, "r-", label="Adalpha Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()


def mnist_ema_w_test(callback,
                     optimizer,
                     epochs=50,
                     learning_rate=0.01,
                     ema_w=[0.99, 0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999],
                     change=0.99,
                     adjustment_exp=2,
                     tests=5):
    """
    Main executable for the program
    :return: None
    """
    adalpha_r_2 = []
    for val in ema_w:
        adalpha_r_2.append(np.mean([adalpha_train_mnist(callback=callback, optimizer=optimizer, epochs=epochs,
                                                        learning_rate=learning_rate, adjustment_exp=adjustment_exp,
                                                        ema_w=val, change=change)[0] for _ in range(tests)], axis=0))

    plt.clf()
    # Graphing the Adalpha Results
    plt.xlabel("ema_w")
    plt.ylabel("Loss")
    plt.title(f"Accuracy vs Ema_w\nOver {epochs} epochs")
    plt.plot(ema_w, adalpha_r_2, "r-", label="Adalpha Acc")
    plt.legend()
    plt.grid(True)
    plt.show()

def mnist_lr_curve(callback,
                   optimizer,
                   epochs=10,
                   learning_rates=[0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
                   ema_w=0.9,
                   change=0.99,
                   adjustment_exp=6,
                   tests=10,
                   copy=False):
    results = []
    for lr in learning_rates:
        results.append(np.mean(mnist_multiple_test(callback=callback, optimizer=optimizer, epochs=epochs,
                                                   learning_rate=lr, ema_w=ema_w, change=change,
                                                   adjustment_exp=adjustment_exp, tests=tests, copy=False), axis=0))

    results = np.asarray(results)
    print(results)
    plt.clf()
    # Graphing the Adalpha Results
    plt.xlabel("lr")
    plt.ylabel("Loss")
    plt.xscale("log")
    plt.title(f"Loss vs Learning Rate\nOver {epochs} epochs on MNIST")
    plt.plot(learning_rates, results[:, 0], "r-", label="Adalpha Loss")
    plt.plot(learning_rates, results[:, 1], "b-", label="Adam Loss")
    plt.legend()
    plt.show()
    if copy:
        pd.DataFrame(results).to_clipboard(excel=True)

def cifar_test(callback, optimizer, epochs=10, learning_rate=0.01, ema_w=0.9, change=0.99, adjustment_exp=6):
    print("Evaluating Adalpha")
    adalpha_acc, adalpha_y_pred, adalpha_y_test = adalpha_train_cifar(callback=callback, optimizer=optimizer,
                                                                      epochs=epochs, learning_rate=learning_rate,
                                                                      adjustment_exp=adjustment_exp, ema_w=ema_w,
                                                                      change=change)
    print("Evaluating Adam")
    acc, y_pred, y_test = adam_train_cifar(epochs, learning_rate)

    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    fig, (ax1, ax2) = plt.subplots(1, 2)

    heatmap_max = make_heatmap(np.argmax(adalpha_y_pred, 1), adalpha_y_test)

    im = ax1.imshow(heatmap_max)

    # Show all ticks and label them with the respective list entries
    ax1.set_xticks(np.arange(len(labels)), labels=labels)
    ax1.set_yticks(np.arange(len(labels)), labels=labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.

    ax1.set_title(f"Predictions vs Actual at\nlr={learning_rate} from Adalpha\nOver {epochs} epochs")

    heatmap = make_heatmap(np.argmax(y_pred, 1), y_test)

    im = ax2.imshow(heatmap)

    # Show all ticks and label them with the respective list entries
    ax2.set_xticks(np.arange(len(labels)), labels=labels)
    ax2.set_yticks(np.arange(len(labels)), labels=labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    ax2.set_title(f"Predictions vs Actual at\nlr={learning_rate} from Adam\nOver {epochs} epochs")
    fig.tight_layout()
    plt.show()
    return adalpha_acc, acc

def cifar_multiple_test(callback, optimizer, epochs=10, learning_rate=0.01, ema_w=0.9, change=0.99, adjustment_exp=6,
                        tests=10, copy=False):
    losses = []
    for i in range(tests):
        losses.append(
            cifar_test(callback=callback, optimizer=optimizer, epochs=epochs, learning_rate=learning_rate,
                       adjustment_exp=adjustment_exp, ema_w=ema_w, change=change))

    if copy:
        pd.DataFrame(losses).to_clipboard(excel=True)
    return losses

def cifar_ema_w_test(callback, optimizer, epochs=50, learning_rate=0.01,
                     ema_w=[0.99, 0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999], change=0.99,
                     adjustment_exp=2):
    """
    Main executable for the program
    :return: None
    """
    adalpha_acc = []
    for val in ema_w:
        adalpha_acc.append(adalpha_train_cifar(callback=callback, optimizer=optimizer, epochs=epochs,
                                               learning_rate=learning_rate, adjustment_exp=adjustment_exp,
                                               ema_w=val, change=change)[0])

    plt.clf()
    # Graphing the Adalpha Results
    plt.xlabel("ema_w")
    plt.ylabel("Loss")
    plt.title(f"Accuracy vs Ema_w\nOver {epochs} epochs on CIFAR-10")
    plt.plot(ema_w, adalpha_acc, "r-", label="Adalpha Acc")
    plt.legend()
    plt.grid(True)
    plt.show()

def cifar_chaos_test(callback, optimizer, epochs=2, learning_rate=0.01, ema_w=0.9, change=0.99,
                     adjustment_exp=[0.2, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]):
    """
    Main executable for the program
    :return: None
    """
    adalpha_r_2 = []
    for val in adjustment_exp:
        adalpha_r_2.append(adalpha_train_cifar(callback=callback, optimizer=optimizer, epochs=epochs,
                                               learning_rate=learning_rate, adjustment_exp=val, ema_w=ema_w,
                                               change=change)[0])

    plt.clf()
    # Graphing the Adalpha Results
    plt.xlabel("Chaos Punishment")
    plt.ylabel("Loss")
    plt.title(f"Accuracy vs Chaos Punishment\nOver {epochs} epochs on CIFAR-10")
    plt.plot(adjustment_exp, adalpha_r_2, "r-", label="Adalpha Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()


def cifar_lr_curve(callback, optimizer, epochs=10,
                   learning_rates=[0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05], ema_w=0.9, change=0.99,
                   adjustment_exp=6, tests=10, copy=False):
    results = []
    for lr in learning_rates:
        results.append(np.mean(cifar_multiple_test(callback=callback, optimizer=optimizer, epochs=epochs,
                                                   learning_rate=lr, ema_w=ema_w, change=change,
                                                   adjustment_exp=adjustment_exp, tests=tests, copy=False), axis=0))

    results = np.asarray(results)
    print(results)
    plt.clf()
    # Graphing the Adalpha Results
    plt.xlabel("lr")
    plt.ylabel("Loss")
    plt.xscale("log")
    plt.title(f"Loss vs Learning Rate\nOver {epochs} epochs on CIFAR")
    plt.plot(learning_rates, results[:, 0], "r-", label="Adalpha Loss")
    plt.plot(learning_rates, results[:, 1], "b-", label="Adam Loss")
    plt.legend()
    plt.show()
    if copy:
        pd.DataFrame(results).to_clipboard(excel=True)

def cartpole_test(callback,
                  optimizer,
                  epochs=50,
                  learning_rate=0.01,
                  ema_w=0.99,
                  change=0.99,
                  adjustment_exp=2,
                  memory_size=10000,
                  cycles=15,
                  tests=10,
                  learning_probability=0.7,
                  learning_size=400,
                  rl_learning_rate=0.2,
                  gamma=0.9,
                  exp_decay=0.995,
                  exploration_rate=0.8,
                  max_steps=500):
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
                         exploration_rate=exploration_rate,
                         max_steps=max_steps)

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
                            exploration_rate=exploration_rate,
                            max_steps=max_steps)

    # Plot the results
    plt.plot(adalpha_results, label="AdAlpha")
    plt.plot(adam_results, label="Adam")
    plt.legend()
    plt.show()

    return adalpha_results, adam_results

def cartpole_multiple_test(callback,
                  optimizer,
                  epochs=50,
                  learning_rate=0.01,
                  ema_w=0.99,
                  change=0.99,
                  adjustment_exp=2,
                  memory_size=1000,
                  cycles=30,
                  rl_tests=10,
                  learning_probability=0.7,
                  learning_size=400,
                  rl_learning_rate=0.2,
                  gamma=0.9,
                  exp_decay=0.995,
                  exploration_rate=0.8,
                  tests=10,
                  copy=True,
                  max_steps=500):
    """
        A function that performs multiple tests of the CartPole environment using a given callback and optimizer.

        Parameters:
            callback (object): The callback function to be used for training.
            optimizer (object): The optimizer object to be used for training.
            epochs (int, optional): The number of epochs to train. Defaults to 50.
            learning_rate (float, optional): The learning rate for the optimizer. Defaults to 0.01.
            ema_w (float, optional): The exponential moving average weight for the optimizer. Defaults to 0.99.
            change (float, optional): The change threshold for the optimizer. Defaults to 0.99.
            adjustment_exp (int, optional): The punishment factor for chaos state. Defaults to 2.
            memory_size (int, optional): The size of the replay memory. Defaults to 10000.
            cycles (int, optional): The number of cycles for training. Defaults to 30.
            rltests (int, optional): The number of tests for reinforcement learning. Defaults to 10.
            learning_probability (float, optional): The probability of learning during training. Defaults to 0.7.
            learning_size (int, optional): The size of the learning set during training. Defaults to 400.
            rl_learning_rate (float, optional): The learning rate for reinforcement learning. Defaults to 0.2.
            gamma (float, optional): The discount factor for reinforcement learning. Defaults to 0.9.
            exp_decay (float, optional): The exponential decay rate for exploration rate. Defaults to 0.995.
            exploration_rate (float, optional): The exploration rate for reinforcement learning. Defaults to 0.8.
            tests (int, optional): The number of tests to perform. Defaults to 10.
            copy (bool, optional): Whether to copy the results to the clipboard. Defaults to True.

        Returns:
            list: A list of losses obtained from each test.
    """

    losses = []
    for i in range(tests):
        adalpha_results, adam_results = cartpole_test(callback=callback,
                                                      optimizer=optimizer,
                                                      learning_rate=learning_rate,
                                                      ema_w=ema_w,
                                                      change=change,
                                                      adjustment_exp=adjustment_exp,
                                                      tests=rl_tests,
                                                      memory_size=memory_size,
                                                      cycles=cycles,
                                                      epochs=epochs,
                                                      learning_probability=learning_probability,
                                                      learning_size=learning_size,
                                                      rl_learning_rate=rl_learning_rate,
                                                      gamma=gamma,
                                                      exp_decay=exp_decay,
                                                      exploration_rate=exploration_rate,
                                                      max_steps=max_steps)
        losses.append([np.mean(adalpha_results), np.max(adalpha_results), np.min(adalpha_results),
                       np.mean(adam_results), np.max(adam_results), np.min(adam_results)])
        if copy:
            pd.DataFrame(np.asarray(losses)[:, [0, 3]]).to_clipboard(excel=True)

    plt.plot(losses, label=["Adalpha Mean", "Adalpha Max", "Adalpha Min", "Adam Mean", "Adam Max", "Adam Min"])
    plt.legend()
    plt.title("CartPole Test Results")
    plt.show()


def cartpole_ema_w_test(callback,
                  optimizer,
                  epochs=50,
                  learning_rate=0.01,
                  ema_w=[0.99],
                  change=0.99,
                  adjustment_exp=2,
                  memory_size=10000,
                  cycles=30,
                  rl_tests=10,
                  learning_probability=0.7,
                  learning_size=400,
                  rl_learning_rate=0.2,
                  gamma=0.9,
                  exp_decay=0.995,
                  exploration_rate=0.8,
                  copy=True):
    results = []
    for weight in ema_w:
        results.append(np.mean(
            adalpha_train_cartpole(callback=callback,
                                                      optimizer=optimizer,
                                                      learning_rate=learning_rate,
                                                      ema_w=weight,
                                                      change=change,
                                                      adjustment_exp=adjustment_exp,
                                                      tests=rl_tests,
                                                      memory_size=memory_size,
                                                      cycles=cycles,
                                                      epochs=epochs,
                                                      learning_probability=learning_probability,
                                                      learning_size=learning_size,
                                                      rl_learning_rate=rl_learning_rate,
                                                      gamma=gamma,
                                                      exp_decay=exp_decay,
                                                      exploration_rate=exploration_rate), axis=0))

    results = np.asarray(results)
    print(results)
    plt.clf()
    # Graphing the Adalpha Results
    plt.xlabel("lr")
    plt.ylabel("Loss")
    plt.xscale("log")
    plt.title(f"Loss vs Learning Rate\nOver {epochs} epochs on Cartpole")
    plt.plot(ema_w, results[:, 0], "r-", label="Adalpha Loss")
    plt.plot(ema_w, results[:, 1], "b-", label="Adam Loss")
    plt.legend()
    plt.show()
    if copy:
        pd.DataFrame(results).to_clipboard(excel=True)

def cartpole_change_test(callback,
                  optimizer,
                  epochs=50,
                  learning_rate=0.01,
                  ema_w=0.99,
                  change=[0.8, 0.99],
                  adjustment_exp=2,
                  memory_size=10000,
                  cycles=30,
                  rl_tests=10,
                  learning_probability=0.7,
                  learning_size=400,
                  rl_learning_rate=0.2,
                  gamma=0.9,
                  exp_decay=0.995,
                  exploration_rate=0.8,
                  copy=True):
    results = []
    for val in change:
        results.append(np.mean(
            adalpha_train_cartpole(callback=callback,
                                                      optimizer=optimizer,
                                                      learning_rate=learning_rate,
                                                      ema_w=ema_w,
                                                      change=val,
                                                      adjustment_exp=adjustment_exp,
                                                      tests=rl_tests,
                                                      memory_size=memory_size,
                                                      cycles=cycles,
                                                      epochs=epochs,
                                                      learning_probability=learning_probability,
                                                      learning_size=learning_size,
                                                      rl_learning_rate=rl_learning_rate,
                                                      gamma=gamma,
                                                      exp_decay=exp_decay,
                                                      exploration_rate=exploration_rate), axis=0))

    results = np.asarray(results)
    print(results)
    plt.clf()
    # Graphing the Adalpha Results
    plt.xlabel("lr")
    plt.ylabel("Loss")
    plt.xscale("log")
    plt.title(f"Loss vs Learning Rate\nOver {epochs} epochs on Cartpole")
    plt.plot(ema_w, results[:, 0], "r-", label="Adalpha Loss")
    plt.plot(ema_w, results[:, 1], "b-", label="Adam Loss")
    plt.legend()
    plt.show()
    if copy:
        pd.DataFrame(results).to_clipboard(excel=True)

def run_all_tests_for_paper():
    mnist_multiple_test(AA.Adalpha_Callback, AA.Adalpha_Momentum, epochs=10, learning_rate=0.001, adjustment_exp=2,
                        ema_w=0.9, change=0.99, copy=True, tests=10)

    cifar_multiple_test(AA.Adalpha_Callback, AA.Adalpha_Momentum, epochs=10, learning_rate=0.0001, adjustment_exp=2,
                        ema_w=0.8, change=0.9, copy=True, tests=50)

    cifar_multiple_test(AA.Adalpha_Callback, AA.Adalpha_Momentum, epochs=10, learning_rate=0.0001, adjustment_exp=2,
                        ema_w=0.9, change=0.99, copy=True, tests=50)

    cartpole_multiple_test(AA.Adalpha_Callback, AA.Adalpha_Momentum, exploration_rate=0.4, epochs=10,
                           learning_rate=0.0001, adjustment_exp=2, ema_w=0.8, change=0.9, cycles=10, copy=True,
                           tests=50, max_steps=500)

    mnist_chaos_test(AA.Adalpha_Callback, AA.Adalpha_Momentum, epochs=10, learning_rate=0.001)

    cifar_chaos_test(AA.Adalpha_Callback, AA.Adalpha_Momentum, epochs=50, learning_rate=0.0001)

    bike_chaos_test(AA.Adalpha_Callback, AA.Adalpha_Momentum, epochs=50)

    mnist_ema_w_test(AA.Adalpha_Callback, AA.Adalpha_Momentum, epochs=50, learning_rate=0.001, adjustment_exp=2,
                     ema_w=[0.8, 0.85, 0.9, 0.95, 0.99])