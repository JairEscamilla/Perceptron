from numpy import exp, array, random, dot # importing functions from numpy library

class Perceptron():
    
    def __init__(self, *args, **kwargs): # Class constructor
        random.seed(1) # Seed of a random generator for the weights inicialization

        self.weights = 2 * random.random((2 , 1)) - 1  # Initializing randomly the weights

    def __sigmoid(self, y): # Activation function
        return 1 / (1 + exp(-y))
    
    def __sigmoid_derivative(self, y): # This is the gradient of the Sigmoid function and indicates the confidence we have in the existing weight.
        return y * (1 - y)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        output = self.think(training_set_inputs)
        error = training_set_outputs = output 

        adjustment = dot(training_set_outputs.T, error * self.__sigmoid_derivative(output))

        self.weights = self.weights + adjustment
    
    def think(self, inputs):
        return self.__sigmoid(dot(inputs, self.weights))

if __name__ == "__main__":
    perceptron = Perceptron()
    training_set_input = array([[0, 0], [0, 1], [1, 0], [1, 1]]) # truth table (inputs)
    training_set_outputs = array([[0, 0, 0, 1]]) # truth table (outputs)

    perceptron.train(training_set_input, training_set_outputs, 10000)

    a = int(input("First value (0/1)-> "))
    b = int(input("Second value (0/1)-> "))

    output = perceptron.think(array([a, b]))

    print("The output of the perceptron is {}".format("1" if output >= 0.68 else "0"))