import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from data_loading import loadData, loadOutlierData
from model import FCNet
from train import TrainModelUpdated


def plot_test_train_losses(hist):
    #  Plotting the MSE loss on test data
    plt.plot(np.mean(hist,-1)[:,1],label="test", color="#495F41")

    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.title("Test Loss Convergence on the updated model with l2 regularization & gradient clipping")
    plt.legend()
    plt.show()

    #  Plotting the MSE loss on training data
    plt.plot(np.mean(hist,-1)[:,0],label="train", color="#FF5666")

    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.title("Training Loss Convergence on the updated model with l2 regularization & gradient clipping")
    plt.legend()
    plt.show()


def run_experiment():

    x,y = loadData()
    plt.scatter(x,y)
    plt.title("Initial data without outliers")
    plt.show()

    print("loading data with outliers....")

    model = FCNet()
    x_train, y_train = loadOutlierData()
    x_test, y_test   = loadData()

    opt    = TrainModelUpdated(model, batch_size = 8, lr = 0.001, 
                            loss = tf.keras.losses.MeanSquaredError, 
                            opt=tf.keras.optimizers.Adam,
                                use_l2_reg=True, use_clipnorm=True,
                                reg_weig_lambda=0.1)

    hist = opt.run(x_train, y_train, x_test, y_test, 30, verbose=1)


    plot_test_train_losses(hist)


if __name__=="__main__":
    run_experiment()