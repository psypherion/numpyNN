from neural_network import NeuralNetwork
import numpy as np

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
