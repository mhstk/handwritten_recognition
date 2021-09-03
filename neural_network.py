# from params import LEARNING_RATE
import numpy as np
import time
import matplotlib.pyplot as plt

"""
A three layered (two hidden layered) Neural Network
"""
class NeuralNetwork:
    def __init__(self, input_num, h1Layer_num, h2Layer_num, out_num, learning_rate, \
                    number_of_epochs, batch_size, activation_method):
        self.input_num = input_num
        self.h2Layer_num = h2Layer_num
        self.h1Layer_num = h1Layer_num
        self.out_num = out_num
        self.learning_rate = learning_rate
        self.number_of_epochs = number_of_epochs
        self.batch_size = batch_size
        self.activation_method, self.diver_activation_method = self.choose_activation_method(activation_method)

        self.w1 = np.ones((self.h1Layer_num, self.input_num))
        self.w2 = np.ones((self.h2Layer_num, self.h1Layer_num))
        self.w3 = np.ones((self.out_num, self.h2Layer_num))

        self.w1 = np.random.normal(0, 0.1, (self.h1Layer_num, self.input_num))
        self.w2 = np.random.normal(0, 0.1, (self.h2Layer_num, self.h1Layer_num))
        self.w3 = np.random.normal(0, 0.1, (self.out_num, self.h2Layer_num))

        self.b1 = np.zeros((self.h1Layer_num , 1))
        self.b2 = np.zeros((self.h2Layer_num , 1))
        self.b3 = np.zeros((self.out_num, 1))


    def set_train_set(self, train_set):
        self.train_set = train_set






    def choose_activation_method(self, activation_str):
        if activation_str == "sigmoid":
            return NeuralNetwork.am_sigmoid, NeuralNetwork.diver_am_sigmoid
        elif activation_str == "relu":
            return NeuralNetwork.am_relu, NeuralNetwork.diver_am_relu
        elif activation_str == "leakyrelu":
            return NeuralNetwork.am_leaky_relu, NeuralNetwork.diver_am_leaky_relu
        elif activation_str == "tanh":
            return NeuralNetwork.am_tanh, NeuralNetwork.diver_am_tanh


    def test(self, inputs):
        start = time.time()

        number_of_true = 0
        for inp, label in inputs:
            outputs = self.forward([(inp, label)])
            number_of_true += NeuralNetwork.calculate_accuracy(outputs)
        
        accuracy = number_of_true/len(inputs)
        print(f"Accuracy for test-set: {accuracy:.4}")

        print() 
        print(f"\ttime spent: {time.time() - start}\n")


    def train(self):
        start = time.time()
        train_set = self.train_set
        costs = []

        for epoch in range(self.number_of_epochs):
            cost = 0
            number_of_true = 0
            for i in range(0,len(train_set), self.batch_size):
                mini_batch = train_set[i:i+self.batch_size]
                outputs = self.forward(mini_batch)

                number_of_true += NeuralNetwork.calculate_accuracy(outputs)
                # print(outputs[0][0])
                # print(outputs[0][1])
                # print(NeuralNetwork.narray_to_one(outputs[0][1]))
                # exit()
                

                cost += self.backpropogation(outputs, i)
            
            cost = cost/len(train_set)
            costs.append(cost)
            accuracy = number_of_true/len(train_set)
            print(f"epoch {epoch+1: <10}Cost: {cost: <28}Accuracy: {accuracy:.4}")

        print() 
        print(f"\ttime spent: {time.time() - start}\n")
        plt.figure("costs in train-set")
        plt.plot(list(range(1,len(costs)+1)), costs)
        plt.ylabel("Cost")
        plt.show()
        

        

    def forward(self, inputs):
        outputs = []
        for inp, y in inputs:
            z1 = self.w1 @ inp + self.b1
            a1 = self.activation_method(z1)
            z2 = self.w2 @ a1 + self.b2
            a2 = self.activation_method(z2)
            z3 = self.w3 @ a2 + self.b3
            a3 = self.activation_method(z3)
            outputs.append((y, a3, z3, a2, z2, a1, z1, inp))
        return outputs
        


    
    def backpropogation(self, outputs, iii):
        cost = 0


        grad_w3 = np.zeros_like(self.w3)
        grad_w2 = np.zeros_like(self.w2)
        grad_w1 = np.zeros_like(self.w1)


        grad_b3 = np.zeros_like(self.b3)
        grad_b2 = np.zeros_like(self.b2)
        grad_b1 = np.zeros_like(self.b1)


        
        for y, a3, z3, a2, z2, a1, z1, inp in outputs:
            cost += np.sum((a3 - y) ** 2)


            div_a3 = 2 * (a3 - y) 
            div_w3 = (div_a3 * self.diver_activation_method(z3)) @ np.transpose(a2) 
            div_b3 = div_a3 * self.diver_activation_method(z3)

            div_a2 = np.transpose(self.w3) @ (div_a3 * self.diver_activation_method(z3))
            div_w2 = (div_a2 * self.diver_activation_method(z2)) @ np.transpose(a1)
            div_b2 = div_a2 * self.diver_activation_method(z2)

            div_a1 = np.transpose(self.w2) @ (div_a2 * self.diver_activation_method(z2))
            div_w1 = (div_a1 * self.diver_activation_method(z1)) @ np.transpose(inp)
            div_b1 = div_a1 * self.diver_activation_method(z1)

            grad_w3 += div_w3
            grad_w2 += div_w2
            grad_w1 += div_w1
            grad_b3 += div_b3
            grad_b2 += div_b2
            grad_b1 += div_b1
            # print("y:\n" , y)
            # print("a3:\n" , a3)
            # print("z3:\n" , z3)
            # print("a2:\n" , a2)
            # print("w3:\n" , self.w3)
            # print("b3:\n", self.b3)
            # print("grad_w3:\n", grad_w3)
            # print("grad_b3:\n", grad_b3)
            # print("\n--------------------------")
            # print("a2:\n" , a2)
            # print("z2:\n" , z2)
            # print("a1:\n" , a1)
            # print("w2:\n" , self.w2)
            # print("b2:\n", self.b2)
            # print("grad_w2:\n", grad_w2)
            # print("grad_b2:\n", grad_b2)
            # print("\n--------------------------")
            # print("a1:\n" , a1)
            # print("z1:\n" , z1)
            # print("a0:\n" , inp)
            # print("w1:\n" , self.w1)
            # print("b1:\n", self.b1)
            # print("grad_w1:\n", grad_w1)
            # print("grad_b1:\n", grad_b1)
            # exit()



        grad_w3 /= len(outputs)
        grad_w2 /= len(outputs)
        grad_w1 /= len(outputs)
        grad_b3 /= len(outputs)
        grad_b2 /= len(outputs)
        grad_b1 /= len(outputs)


        self.w3 -= self.learning_rate * grad_w3
        self.w2 -= self.learning_rate * grad_w2
        self.w1 -= self.learning_rate * grad_w1
        self.b3 -= self.learning_rate * grad_b3
        self.b2 -= self.learning_rate * grad_b2
        self.b1 -= self.learning_rate * grad_b1

        return cost
        


    @staticmethod
    def calculate_accuracy(outputs):
        trues = 0
        for i in range(len(outputs)):
            if np.array_equal(NeuralNetwork.narray_to_one(outputs[i][1]), outputs[i][0]):
                trues += 1
        return trues




    @staticmethod
    def narray_to_one(narry):
        zeros = np.zeros_like(narry)
        zeros[narry.argmax()] = 1
        return zeros

    @staticmethod
    def am_sigmoid(matrix):
        return 1/(1 + np.exp(-1 * matrix))

    @staticmethod
    def diver_am_sigmoid(matrix):
        y = NeuralNetwork.am_sigmoid(matrix)
        return y * (1-y)

    @staticmethod
    def am_relu(matrix):
        return ((matrix > 0) * matrix)

    @staticmethod
    def diver_am_relu(matrix):
        return ((matrix > 0) * 1)

    @staticmethod
    def am_leaky_relu(matrix):
        y1 = ((matrix > 0) * matrix)
        y2 = ((matrix <= 0) * matrix * 0.01)
        return y1 + y2

    @staticmethod
    def diver_am_leaky_relu(matrix):
        y1 = ((matrix > 0) * 1)
        y2 = ((matrix <= 0) * 0.01)
        return y1 + y2

    @staticmethod
    def am_tanh(matrix):
        t=(np.exp(matrix)-np.exp(-matrix))/(np.exp(matrix)+np.exp(-matrix))
        return t

    @staticmethod
    def diver_am_tanh(matrix):
        t= NeuralNetwork.am_tanh(matrix)
        dt=1-t**2
        return dt






