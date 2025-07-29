
import numpy as np


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


class Neuron:
    def __init__(self): #初期設定
        self.sum = 0.0
        self.output = 0.0

    def set_input(self, input):
        self.sum += input

    def get_output(self):
        self.output = sigmoid(self.sum)
        return self.output


class NeuralNetwork:
    def __init__(self):
        self.neuron = Neuron()
        self.weight = [ 1.5, 0.75, -1.0]
        self.bias = 1.0
    
    def commit(self, inputs):
        for i in range( 0, len(inputs) ):
            self.neuron.set_input(inputs[i] * self.weight[i])
        self.neuron.set_input(self.bias)
        return self.neuron.get_output()

network = NeuralNetwork()

inputs = [1.0, 2.0, 3.0]

print( network.commit(inputs) )

