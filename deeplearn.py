'''
This module allows access to an arbitrarily large connected neural network
Matrix notation = (3, 4) is 3 rows, 4 columns
'''
import numpy as np 
from time import sleep

class NeuralNetwork(object):
    def __init__(self, size=None, connections=None):
        '''
        Size should be a tuple with layer sizes: (3, 4, 4, 5, 3, 2) = 3 feature inputs, 4 hidden layers, 2 outputs || 6 total layers with 5 synapses
        L0  L1  L2  L3  L4  L5
        *   *   *   *   *   *

        *   *   *   *   *   *

        *   *   *   *   *  

            *   *   *

                    *

        '''
        self._reluDeriv = np.vectorize(self._relu_Deriv_nonvec)

        if connections == None:
            self._testSize(size)
            self.size = size 
            self.connections = self._createConnections(self.size)

        else:
            self._testConnections(connections)
            self.connections = connections
            self.size = self._determineSize(connections)

        self._default_activation = self.sigmoid
        self._layer_out = []
        self._layer_deltas = []

    #
    # Data Model Functions "dunder"
    #

    def __repr__(self):
        rep = "NeuralNetwork({})".format(str(self.size))
        return rep


    def __str__(self):
        layer_sizes = 'Network Connections: {} | '.format(len(self.connections))
        for index, connection in enumerate(self.connections):
            if index < len(self.connections) - 1:
                layer_sizes += '({}, {}), '.format(len(connection), len(connection[0]))
            else:
                layer_sizes += '({}, {})'.format(len(connection), len(connection[0]))

        return layer_sizes

    #
    # Network Setup Internal Functions
    #

    def _testConnections(self, connections):
        test_array = np.array([1])
        test_array2 = np.array([1.1])

        if type(connections) != list:
            raise TypeError("Connections must be a list")

        for connection in connections:
            if type(connection) != type(test_array):
                raise TypeError("Connections must be a list of numpy arrays")

        for synapse in connections:
            for row in synapse:
                for col in row:
                    if type(col) != type(test_array[0]) and type(col) != type(test_array2[0]):
                        print(col, type(col))
                        print(test_array[0], type(test_array[0]))
                        raise TypeError("All values in connections must be floats or integers")


    def _testSize(self, size):
        '''Ensures that the size designated is a list or tuple and that only integers were given'''
        if type(size) != list and type(size) != tuple:
            raise TypeError("Size must be a tuple or a list")
        else:
            for num in size:
                if type(num) != int:
                    raise TypeError("Layer sizes must be integers: {}".format(str(num)))


    def _determineSize(self, connections):
        size_determination = []

        for index, connection in enumerate(connections):
            if index < len(connections) - 1:
                size_determination.append(len(connection) - 1)
            else:
                size_determination.append(len(connection) - 1)
                size_determination.append(len(connection[0]))

        return size_determination


    def _createConnections(self, size):
        '''Walks through the size variable and adds connections, +1 expected inputs for each layer to account for bias'''
        connections = []
        for l0, l1 in zip(self.size[:-1], self.size[1:]):
            layer = np.random.random((l0 + 1, l1)) * 2 - 1
            connections.append(layer)

        return connections

    #
    # Internal Training Functions
    #

    def _addBias(self, layer):
        '''Adds a 1 to the end of each row'''
        rows = len(layer)
        bias = np.ones((rows, 1))

        with_bias = np.hstack((layer, bias))

        return with_bias

    def _removeBias(self):
        ''' Will need to remove bias eventually'''
        pass

    #
    # Activation Functions
    #

    def softMax(self, layer):
        '''Forces a neuron to only fire the most confident output'''
        for row in layer:
            largest = 0.0
            index_max = 0

            for index, col in enumerate(row):
                if col > largest:
                    largest = col
                    index_max = index
                    print(index_max)

            print("overall largest {}".format(str(largest)))
            for index in range(len(row)):
                if index == index_max:
                    row[index] = 1
                else:
                    row[index] = 0

        return layer


    def sigmoid(self, z, deriv=False):
        '''Squashes outputs between 0 and 1'''
        output = 1 / (1 + np.exp(-z))

        if not deriv:
            return output
        else:
            return output * (1 - output)


    def _relu_Deriv_nonvec(self, z):
        if z > 0:
            return 1
        else:
            return 0


    def relu(self, z, deriv=False):
        '''Forces 0 or 1 for outside of domain(-1, 1), else y = x + 1'''
        output = np.maximum(z, 0, z)

        if not deriv:
            return output
        else: # this is not going to work because you can't do a comparison operator on an entire nump array, need to look up how to do this element wise
            return self._reluDeriv(z)


    def softPlus(self, z, deriv=False):
        '''A smooth approximation of the relu activation function'''
        output = np.log(1 + np.exp(z))

        if not deriv:
            return output
        else:
            return self.sigmoid(z)

    #
    # Training Logic: Forward Propagation
    #

    def forwardPass(self, inpts, activation):
        '''One forward propagation through the network'''
        if activation == None:
            activation = self._default_activation

        self._layer_out = []

        for index, connection in enumerate(self.connections):
            if index == 0:
                with_bias = self._addBias(inpts)
                output = np.dot(with_bias, connection)
                activated = activation(output)
                self._layer_out.append(activated)

            else:
                with_bias = self._addBias(self._layer_out[-1])
                output = np.dot(with_bias, connection)
                activated = activation(output)
                self._layer_out.append(activated)


    def backwardPass(self, labels, activation=None):
        '''One backwards propagation through the network'''
        if activation == None:
            activation = self._default_activation

        self._layer_deltas = []

        for index, output in enumerate(reversed(self._layer_out)):
            if index == 0: #very last output
                error = labels - output
                delta = error * activation(output, deriv=True)
                self._layer_deltas.append(delta)

            else:
                error = np.dot(self._layer_deltas[-1], self.connections[index * -1].T) 
                # Since the connections expect an additional feature in the inputs (additional column) as a bias, they have an additional row
                # When you transpose the synapse here (above), it will lead to the error having an extra col compared to the activation(output)
                error = error[:,:-1] #every row, all but last column (pertaining to the bias)
                delta = error * activation(output, deriv=True)
                self._layer_deltas.append(delta)


    def updateWeights(self, inpts, LR, LR_Decay):
        '''Uses the delta and layer output matrix to update the appropriate weights'''
        if LR_Decay != None:
            LR = LR_Decay(LR)

        for index, connection in enumerate(self.connections):
            if index == 0:
                withBias = self._addBias(inpts)
                connection += np.dot(withBias.T, self._layer_deltas[(index + 1) * - 1]) * LR #loop over the deltas in reverse
            else:
                withBias = self._addBias(self._layer_out[index - 1])
                connection += np.dot(withBias.T, self._layer_deltas[(index + 1) * - 1]) * LR


    def train(self, inpts, labels, epocs=1000, LR=0.05, LR_Decay=None, activation=None, finalLayerActivation=None):
        for _ in range(epocs):
            self.forwardPass(inpts, activation)
            self.backwardPass(labels, activation)
            self.updateWeights(inpts, LR, LR_Decay)

        if finalLayerActivation == None:
            print("Final Output:\n", self._layer_out[-1])
        else:
            out = self.finalLayerActivation(self._layer_out[-1])
            print("Final Output:\n", out)



if __name__ == '__main__':
    brain = NeuralNetwork(size=(3, 2, 3, 1))
    inp = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 0]])
    labels = np.array([[1], [0], [0], [0]])

    brain.train(inp, labels, epocs=100000, activation=brain.relu)

    sleep(10)
