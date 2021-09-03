import matplotlib.pyplot as plt
from read_mnist_dataset import read_image_label, show_image
from neural_network import NeuralNetwork
from params import *



if __name__ == "__main__":
    print()
    print("-Reading train-set...")
    train_set = read_image_label(TRAIN_SET_IMAGES, TRAIN_SET_LABELS)
    nn = NeuralNetwork(784, 16, 16, 10, **nn_paramaters)
    nn.set_train_set(train_set)
    print("-Training with dataset...")
    nn.train()

    print("\n----------------------------------------------------------------------------------------------------------------------")
    print("-Reading test-set...")
    test_set = read_image_label(TEST_SET_IMAGES, TEST_SET_LABELS)
    print("-Testing...")
    nn.test(test_set)






    

