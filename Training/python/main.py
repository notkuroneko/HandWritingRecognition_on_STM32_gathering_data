import numpy as np 

np.random.seed(0)

'''
Function to initialize weights
	Take the shape (n_inputs, n_neurons) of a layer and 
	return randomize weight with the same shape 
'''
def initialize_weights(shape):
    return np.random.randn(*shape) * 0.10


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    # Forawrd pass
    def forward(self, inputs):
    	self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
   	
    # Backward pass
   	def backward(self, dvalues):
   		# Gradients on parameters
   		self.dweights = np.dot(self.input.T, dvalues)
   		self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
   		# Gradient on values
   		self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

    def backpropagation(x):
    	return (x > 0).astype(float)


class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


