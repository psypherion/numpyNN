import numpy as np
from warnings import filterwarnings

class NeuralNetwork:
    def __init__(self):
        self.weights_list = []
        self.bias_list = []
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def dense(self, ip, weights, bias):
        units = weights.shape[1]
        weight = [weights[:, i] for i in range(units)]
        g = np.empty((1, units))  # Initialize an empty array for output
        
        for i in range(units):
            z = np.dot(ip, weight[i]) + bias[i]
            g[0, i] = self.sigmoid(z)[0]
        
        return g
    
    def add_layer(self, weights, bias):
        self.weights_list.append(weights)
        self.bias_list.append(bias)
    
    def layers(self, n, ip):
        if n == 0:
            return ip  
        else:
            output = self.dense(ip, self.weights_list[len(self.weights_list) - n], self.bias_list[len(self.bias_list) - n])
            return self.layers(n - 1, output)


if __name__ == "__main__":
    # Create an instance of the NeuralNetwork class
    network = NeuralNetwork()

    # Define the weights and biases for each layer
    weights = [
        np.array([[1, -3, 5], [2, 4, -6]]),   # Weights for the first layer (2x3)
        np.array([[-1, 1, 2, 7], [1, 2, 3, -3], [1, 2, 3, 4]]),  # Weights for the second layer (2x4)
        np.array([[-1, 6], [1, -2], [1, 2], [1, 3]])  # Weights for the third layer (2x2)
    ]

    biases = [
        np.array([-1, 1, 2]),  # Biases for the first layer (1x3)
        np.array([1, 2, 3, 4]),  # Biases for the second layer (1x4)
        np.array([1, 2])  # Biases for the third layer (1x2)
    ]

    # Add layers to the network
    for w, b in zip(weights, biases):
        network.add_layer(w, b)

    # Example input data
    input_data = np.array([[200.0, 17.0]])

    # Call the layers method to get the final output
    final_output = network.layers(len(weights), input_data)

    print("Final Output:", final_output)
