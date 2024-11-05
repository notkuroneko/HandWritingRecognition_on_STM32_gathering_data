import numpy as np

# Initialize weights
def initialize_weights(shape):
    return np.random.randn(*shape) * 0.1

def relu_derivative(x):
    return (x > 0).astype(float)

# Softmax function for output layer
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)

# Convolution Layer
def convolution(image, kernel):
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1
    output = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            output[i, j] = np.sum(image[i:i+kernel_height, j:j+kernel_width] * kernel)
    
    return output

# Max Pooling Layer
def max_pooling(feature_map, size=2, stride=2):
    output_height = feature_map.shape[0] // stride
    output_width = feature_map.shape[1] // stride
    pooled_output = np.zeros((output_height, output_width))

    for i in range(0, feature_map.shape[0], stride):
        for j in range(0, feature_map.shape[1], stride):
            pooled_output[i // stride, j // stride] = np.max(feature_map[i:i+size, j:j+size])
    
    return pooled_output

# Fully Connected Layer
def fully_connected(input_data, weights, biases):
    return np.dot(weights, input_data) + biases

# Flatten Layer
def flatten(feature_maps):
    return feature_maps.flatten()

# Forward Pass
def forward(image, params):
    # Convolution Layer
    conv_output = convolution(image, params['conv_kernel'])
    conv_output = relu(conv_output)
    
    # Pooling Layer
    pooled_output = max_pooling(conv_output)
    
    # Flatten Layer
    flat_output = flatten(pooled_output)
    
    # Fully Connected Layer
    fc_output = fully_connected(flat_output, params['fc_weights'], params['fc_bias'])
    
    # Softmax Layer (Output)
    output = softmax(fc_output)
    
    return conv_output, pooled_output, flat_output, fc_output, output

# Loss Function: Mean Squared Error
def compute_loss(predictions, labels):
    return np.mean((predictions - labels) ** 2)

# Backward Pass
def backward(image, conv_output, pooled_output, flat_output, fc_output, output, label, params, lr=0.01):
    # Loss gradient with respect to softmax output
    output_error = output - label
    
    # Fully Connected Layer Gradients
    fc_weights_grad = np.outer(output_error, flat_output)
    fc_bias_grad = output_error
    
    # Gradient through ReLU and Pooling
    flat_output_error = np.dot(params['fc_weights'].T, output_error)
    pooled_output_error = flat_output_error.reshape(pooled_output.shape)
    
    # Backpropagate Max Pooling (Gradient of Max is passed only to max value locations)
    conv_output_error = np.zeros_like(conv_output)
    stride = 2
    for i in range(0, pooled_output.shape[0]):
        for j in range(0, pooled_output.shape[1]):
            h, w = i * stride, j * stride
            pool_region = conv_output[h:h+2, w:w+2]
            max_val = np.max(pool_region)
            conv_output_error[h:h+2, w:w+2] = (pool_region == max_val) * pooled_output_error[i, j]

    # Backpropagate Convolution
    kernel_gradient = np.zeros_like(params['conv_kernel'])
    for i in range(conv_output_error.shape[0]):
        for j in range(conv_output_error.shape[1]):
            kernel_gradient += conv_output_error[i, j] * image[i:i+params['conv_kernel'].shape[0], j:j+params['conv_kernel'].shape[1]]
    
    # Update weights and biases
    params['fc_weights'] -= lr * fc_weights_grad
    params['fc_bias'] -= lr * fc_bias_grad
    params['conv_kernel'] -= lr * kernel_gradient

# Training Loop
def train(images, labels, params, epochs=10, lr=0.01):
    for epoch in range(epochs):
        total_loss = 0
        for image, label in zip(images, labels):
            # Forward pass
            conv_output, pooled_output, flat_output, fc_output, output = forward(image, params)
            
            # Loss calculation
            loss = compute_loss(output, label)
            total_loss += loss
            
            # Backward pass
            backward(image, conv_output, pooled_output, flat_output, fc_output, output, label, params, lr)
        
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(images)}")

# Parameters initialization
params = {
    'conv_kernel': initialize_weights((3, 3)),
    'fc_weights': initialize_weights((10, 13 * 13)),  # Adjust shape based on input size
    'fc_bias': np.zeros(10)
}

# Example usage with random data (replace with real images and labels)
images = np.random.randn(5, 28, 28)  # Example images
labels = np.eye(10)[np.random.choice(10, 5)]  # Example one-hot labels

train(images, labels, params, epochs=5, lr=0.01)
