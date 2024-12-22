/*
	File: cnn.c
	Author: Tran Ba Thanh
	Created: 21/12/2024
	Last update: 21/12/2024
	Purpose: to impliment trained weight from python in C

	Todo:
		
	Model in Python:
		class moderu(torch.nn.Module):
			def __init__(self):
				super(moderu,self).__init__()
				self.conv1 = torch.nn.Conv2d(1,5,3) #input: 1 28x28 channel, kernel 3x3 -> output: 5 features for 2 26x26 channels
				self.conv2 = torch.nn.Conv2d(5,10,3) #input: 5 26x26 channels, kernel 3x3 -> output: 10 features for 20 24x24 channels (maxpooling 2x2 -> 10 12x12 channels)
				self.conv3 = torch.nn.Conv2d(10,15,3) #input: 10 12x12 channels, kernel 3x3 -> output: 15 features for 30 10x10 channels
				self.conv4 = torch.nn.Conv2d(15,20,3) #input: 15 10x10 channels, kernel 3x3 -> output: 20 features for 40 8x8 channels (maxpooling 2x2 -> 20 4x4 channels)
				self.lin1 = torch.nn.Linear(20*4*4,50) #fully connected layer, 50 nodes
				self.lin2 = torch.nn.Linear(50,10) #fully connected layer, 10 nodes (for 10 labels)
			def forward(self,x):
				x = F.relu(self.conv1(x))
				x = F.max_pool2d(F.relu(self.conv2(x)),2)
				x = F.relu(self.conv3(x))
				x = F.max_pool2d(F.relu(self.conv4(x)),2)
				x = x.view(-1, self.num_flat_features(x))
				x = F.relu(self.lin1(x))
				x = F.softmax(self.lin2(x), dim = 1)
				return x
			def num_flat_features(self,x):
				size = x.size()[1:]  # all dimensions except the batch dimension
				num_features = 1
				for s in size:
					num_features *= s
				return num_features
	
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#define __FLT_MAX__ (float)340282300000000000000000000000000000000

// #define LOL lol
#define INPUT_SIZE 28
#define FILTER_SIZE 3
#define AFTER_KERNEL_CAL(N) (N - FILTER_SIZE + 1)	// P = 0; S = 1
#define OUTPUT_SIZE 10		// Number of output class, in this case 10 (0 to 9)

typedef struct {
	float weights[FILTER_SIZE][FILTER_SIZE];
	float bias;
} Filter;

// function prototype
void read_input_data(const char *filename, float input[INPUT_SIZE][INPUT_SIZE]);
float apply_filter(float *input, Filter *filter, int x, int y) ;
void conv2d(float *input_mat, float *output_mat, Filter *filter, int input_mat_size);
void relu(float *mat, int mat_size);
void softmax(float *input_mat, float *output_mat, int mat_size);
void fullyconnected(float *input_mat, float *output_mat, float *weight, float *bias, int input_mat_size, int output_mat_size);
void maxpooling(float *input_mat, float *output_mat, int input_mat_size, int pooling_size);


int main()
{
	// input
	float input_mat[28][28] = {0};
	float *layer1ch0_mat, *layer1ch1_mat, *layer1ch2_mat, *layer1ch3_mat, *layer1ch4_mat;	// 4 channels of layer 1
	// float *layer2_z, *layer2_a;

	// load filter
	Filter filter_conv1;
	filter_conv1.weight = {0};
	filter_conv1.bias = 0;
	// Filter filter_conv2;
	// Filter filter_conv3;
	// Filter filter_conv4;

	// fullyconnected weight and bias
	float fc_weight[][] = {0};
	float fc_bias[] = {0};

	// Calculation
		// layer 1
	layer1ch0_mat = (float *)malloc(sizeof(input_mat));
	conv2d(&input_mat[0][0], layer1ch0_mat, filter_conv1, 28);
	relu(layer1ch0_mat, 28);

	return 0;
}

// Function to read input data from a file
void read_input_data(const char *filename, float input[INPUT_SIZE][INPUT_SIZE]) {
	FILE *file = fopen(filename, "r");
	if (!file) {
		perror("Failed to open file");
		exit(EXIT_FAILURE);
	}
	for (int i = 0; i < INPUT_SIZE; i++) {
		for (int j = 0; j < INPUT_SIZE; j++) {
			fscanf(file, "%f", &input[i][j]);
		}
	}
	fclose(file);
}


/*	Function: apply_filter - applying kernel in conv layer
	Input:
		input: input matrix
		filter: kernel
		x, y: coordinates of destinaiton matrix
	Output:
		sum = total-sigma(input * weight) + bias
*/
float apply_filter(float *input, Filter *filter, int x, int y) 
{
	float sum = 0.0;
	for (int i = 0; i < FILTER_SIZE; i++)			// for accessing row of mat
	{
		for (int j = 0; j < FILTER_SIZE; j++) 		// for accessing col of mat
		{
			sum += input[x + i][y + j] * filter->weights[i][j];
		}
	}
	return sum + filter->bias; // Add bias
}

/*	Function: conv2d - convolution 2d of an layer
	Input:
	Output:
*/
void conv2d(float *input_mat, float *output_mat, Filter *filter, int input_mat_size) 
{
	for (int i = 0; i < AFTER_KERNEL_CAL(input_mat_size); i++)
	{
		for (int j = 0; j < AFTER_KERNEL_CAL(input_mat_size); j++)
		{
			output[i][j] = apply_filter(input, filter, i, j);
		}
	}
}

/*	Function: relu
*/
void relu(float *mat, int mat_size) 
{
	for (int i = 0; i < mat_size; i++) 
	{
		for (int j = 0; j < mat_size; j++) 
		{
			if (mat[i][j] < 0) 
			{
				mat[i][j] = 0;
			}
		}
	}
}


/*	Function: softmax - 
	Input:
	Output:
*/
void softmax(float *input_mat, float *output_mat, int mat_size)
{
	float total_exp = 0.0;
	// calculate sum of e**input_mat[i]
	for (int i = 0; i < mat_size; i++)
	{
		total_exp += exp(input_mat[i]);
	}
	// calculate softmax result
	for (int i = 0; i < mat_size; i++)
	{
		output_mat[i] = exp(input_mat[i]) / total_exp;
	}
}

/*	Function: fullyconnected
	Input:
	Output:
*/
void fullyconnected(float *input_mat, float *output_mat, float *weight, float *bias, int input_mat_size, int output_mat_size)
{
	// need to impliment matmul, but too reiji so ...
	// loop though the output mat to save cal result
	for (int i = 0; i < output_mat_size; i++)			// i for output_mat index
	{
		output_mat[i] = 0;
		// loop though the input mat to cal
		for (int j = 0; j < input_mat_size; j ++)		// j for input_mat index
		{
			output_mat[i] += input_mat[j] * weight[i][j];
		}
		output_mat[i] += bias[i]; 						// adding bias
	}
}

/*
*/
void maxpooling(float *input_mat, float *output_mat, int input_mat_size, int pooling_size)
{
	 // Calculate the size of the output matrix
	int output_mat_size = (int)(input_mat_size / pooling_size);

	// Iterate through the output matrix
	for (int i = 0; i < output_mat_size; i++) 
	{
		for (int j = 0; j < output_mat_size; j++)
		{
			float max_value = -__FLT_MAX__; // Initialize to the smallest float value
			
			// Iterate through the pooling window
			for (int m = 0; m < pooling_size; m++) 
			{
				for (int n = 0; n < pooling_size; n++) 
				{
					// Calculate the index of the current input element
					int row_idx = i * pooling_size + m;
					int col_idx = j * pooling_size + n;
					// int input_idx = row_idx * input_mat_size + col_idx;

					// Update the maximum value
					if (input_mat[input_idx] > max_value) {
						// max_value = input_mat[input_idx];
						max_value = input_mat[row_idx][col_idx];

					}
				}
			}

			// Assign the maximum value to the output matrix
			// int output_idx = i * output_mat_size + j;
			// output_mat[output_idx] = max_value;
			output_mat[i][j] = max_value;
		}
	}
}

// void matmul_a_b()
// void matmul_a_bt()
// void matmul_at_b()
/*
	Reference:
		https://github.com/usamahz/cnn.git
		https://youtu.be/jDe5BAsT2-Y?si=i9AY3u8cukuy75CI
		
*/