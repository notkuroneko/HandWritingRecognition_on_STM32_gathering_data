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
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "cnn.h"
#include "constants.h"
#include "conv1_params.h"
#include "conv2_params.h"
// #include "conv3_params.h"
// #include "conv4_params.h"

#define DEBUG 0

// size = 28*28*1*4 = 3136 bytes
const float input_mat_temp[INPUT_SIZE][INPUT_SIZE] = {
	    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	    {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	    {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	    {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	    {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	    {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	    {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	    {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	    {0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
	    {0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
	    {0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
	    {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
	    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
};

int main()
{
// variable and stuff
	// input matrix data
	float input_mat[1][28][28];
	memcpy(&input_mat[0][0][0], input_mat_temp, sizeof(input_mat_temp));
	
	// layer 1 stuff - l1~~~
	const int l1_ch = NUM_05CH;
	const int l1_dim = 26;
	float l1_mat[l1_ch][l1_dim][l1_dim];	// mat to store calculated value
	int l1_mat_bytesize = sizeof(float)*l1_ch*l1_dim*l1_dim;
	memset(l1_mat, 0, l1_mat_bytesize);
	// initialize filter
	Filter conv1_filter[l1_ch];
		// bias
	load_bias(conv1_filter, conv1_filter_bias, NUM_05CH);

		// weights
	load_weights(conv1_filter[0].weights[0], conv1_filter_ch00_weights00);
	load_weights(conv1_filter[1].weights[0], conv1_filter_ch01_weights00);
	load_weights(conv1_filter[2].weights[0], conv1_filter_ch02_weights00);
	load_weights(conv1_filter[3].weights[0], conv1_filter_ch03_weights00);
	load_weights(conv1_filter[4].weights[0], conv1_filter_ch04_weights00);

	#ifdef DEBUG
	printf("\n=======================================================================\n");
	printf("Print value of conv1_filter[].bias\n");
	printFilterBias(conv1_filter, NUM_05CH);
	printf("Print value of 5ch of conv1_filter[].weights[0]\n");
	printMatrix((float *)conv1_filter[0].weights[0], 5, FILTER_SIZE);
	#endif


	// layer 2 stuff - l1~~~
	const int l2_ch = 10;
	const int l2_dim = 24;
		// mat to store conv2 calculated value
	float l2_mat[l2_ch][l2_dim][l2_dim];	
	int l2_mat_bytesize = sizeof(float)*l2_ch*l2_dim*l2_dim;
	memset(l2_mat, 0, l2_mat_bytesize);
		// mat to store maxpooling calculated value
	const int l2_pool_mat_dim = 12; 		// l2_pool_mat_dim = l2_dim / 2
	float l2_pool_mat[l2_ch][l2_pool_mat_dim][l2_pool_mat_dim];
	int l2_pool_mat_bs = sizeof(float)*l2_ch*l2_pool_mat_dim*l2_pool_mat_dim;
	memset(l2_pool_mat, 0, l2_pool_mat_bs);
		// Filter & init filter
	Filter conv2_filter[l2_ch];
			// bias
	load_bias(conv2_filter, conv2_filter_bias, NUM_10CH);
			// weights
				// ch00
	load_weights(conv2_filter[0].weights[0], conv2_filter_ch00_weights00);
	load_weights(conv2_filter[0].weights[1], conv2_filter_ch00_weights01);
	load_weights(conv2_filter[0].weights[2], conv2_filter_ch00_weights02);
	load_weights(conv2_filter[0].weights[3], conv2_filter_ch00_weights03);
	load_weights(conv2_filter[0].weights[4], conv2_filter_ch00_weights04);
				// ch01
	load_weights(conv2_filter[1].weights[0], conv2_filter_ch01_weights00);
	load_weights(conv2_filter[1].weights[1], conv2_filter_ch01_weights01);
	load_weights(conv2_filter[1].weights[2], conv2_filter_ch01_weights02);
	load_weights(conv2_filter[1].weights[3], conv2_filter_ch01_weights03);
	load_weights(conv2_filter[1].weights[4], conv2_filter_ch01_weights04);
				// ch02
	load_weights(conv2_filter[2].weights[0], conv2_filter_ch02_weights00);
	load_weights(conv2_filter[2].weights[1], conv2_filter_ch02_weights01);
	load_weights(conv2_filter[2].weights[2], conv2_filter_ch02_weights02);
	load_weights(conv2_filter[2].weights[3], conv2_filter_ch02_weights03);
	load_weights(conv2_filter[2].weights[4], conv2_filter_ch02_weights04);
				// ch03
	load_weights(conv2_filter[3].weights[0], conv2_filter_ch03_weights00);
	load_weights(conv2_filter[3].weights[1], conv2_filter_ch03_weights01);
	load_weights(conv2_filter[3].weights[2], conv2_filter_ch03_weights02);
	load_weights(conv2_filter[3].weights[3], conv2_filter_ch03_weights03);
	load_weights(conv2_filter[3].weights[4], conv2_filter_ch03_weights04);
				// ch04
	load_weights(conv2_filter[4].weights[0], conv2_filter_ch04_weights00);
	load_weights(conv2_filter[4].weights[1], conv2_filter_ch04_weights01);
	load_weights(conv2_filter[4].weights[2], conv2_filter_ch04_weights02);
	load_weights(conv2_filter[4].weights[3], conv2_filter_ch04_weights03);
	load_weights(conv2_filter[4].weights[4], conv2_filter_ch04_weights04);
				// ch05
	load_weights(conv2_filter[5].weights[0], conv2_filter_ch05_weights00);
	load_weights(conv2_filter[5].weights[1], conv2_filter_ch05_weights01);
	load_weights(conv2_filter[5].weights[2], conv2_filter_ch05_weights02);
	load_weights(conv2_filter[5].weights[3], conv2_filter_ch05_weights03);
	load_weights(conv2_filter[5].weights[4], conv2_filter_ch05_weights04);
				// ch06
	load_weights(conv2_filter[4].weights[0], conv2_filter_ch06_weights00);
	load_weights(conv2_filter[4].weights[1], conv2_filter_ch06_weights01);
	load_weights(conv2_filter[4].weights[2], conv2_filter_ch06_weights02);
	load_weights(conv2_filter[4].weights[3], conv2_filter_ch06_weights03);
	load_weights(conv2_filter[4].weights[4], conv2_filter_ch06_weights04);

	#ifdef DEBUG
	printf("\n=======================================================================\n");
	printf("Print value of conv2_filter[].bias\n");
	printFilterBias(conv2_filter, NUM_10CH);
	printf("Print value of 5ch of conv2_filter[0].weights[]\n");
	printMatrix((float *)conv1_filter[0].weights[0], 5, FILTER_SIZE);
	printf("Print value of 5ch of conv2_filter[1].weights[]\n");
	printMatrix((float *)conv1_filter[1].weights[0], 5, FILTER_SIZE);
	printf("Print value of 5ch of conv2_filter[2].weights[]\n");
	printMatrix((float *)conv1_filter[2].weights[0], 5, FILTER_SIZE);
	printf("Print value of 5ch of conv2_filter[3].weights[]\n");
	printMatrix((float *)conv1_filter[3].weights[0], 5, FILTER_SIZE);
	printf("Print value of 5ch of conv2_filter[4].weights[]\n");
	printMatrix((float *)conv1_filter[4].weights[0], 5, FILTER_SIZE);
	#endif

	// layer 3 stuff
	// const int l3_ch = 15;
	// const int l3_dim = 10;
	// float l3_mat[l3_ch][l3_dim][l3_dim];	// mat to store calculated value
	// int l3_mat_bs = sizeof(float)*l3_ch*l3_dim*l3_dim;
	// memset(l3_mat, 0, l3_mat_bs);
	// Filter conv3_filter[l3_ch] = {
	//     {   // ch00
	//         .weights = {
	//             { 0.0069994, -0.10749,  0.11241},
	//             { -0.12853, -0.082408,  0.099403},
	//             { 0.10726, -0.082765, -0.11805}
	//         },
	//         .bias = 0.0311
	//     },
	//     {	//ch02

	//     },
    // };

	// layer 4 stuff
	// int l4_ch = 20;
	// int l4_dim = 8;

	// fullyconnected weight and bias
	// float fc_weight[][] = {0};
	// float fc_bias[] = {0};

// Calculation
	// layer 1
	// void conv2d(float *in_mat, float *out_mat, Filter *filter, int in_dim, int in_ch, int out_ch)
	conv2d((float *)input_mat, (float *)l1_mat, conv1_filter, 28, NUM_01CH, NUM_05CH);
	relu((float *)l1_mat, 26, NUM_05CH);

	// layer 2
	// conv2d(l1_mat, l2_mat, conv2_filter, 26, NUM_05CH, NUM_10CH);
	// relu(l2_mat, 24, NUM_10CH);
	// maxpooling(l2_mat, l2_pool_mat, 24, 2, NUM_10CH);

	printf("\n=======================================================================\n");
	printMatrix((float *)l1_mat, NUM_05CH, 26);

	return 0;
}

// Function to read input data from a file
// void read_input_data(const char *filename, float input[INPUT_SIZE][INPUT_SIZE]) {
// 	FILE *file = fopen(filename, "r");
// 	if (!file) {
// 		perror("Failed to open file");
// 		exit(EXIT_FAILURE);
// 	}
// 	for (int i = 0; i < INPUT_SIZE; i++) {
// 		for (int j = 0; j < INPUT_SIZE; j++) {
// 			fscanf(file, "%f", &input[i][j]);
// 		}
// 	}
// 	fclose(file);
// }


/*	Function: apply_filter - applying kernel in conv layer
	Input:
		input: input matrix
		filter: kernel
		x, y: coordinates of destinaiton matrix
	Output:
		sum = total-sigma(input * weight) + bias
*/
// float apply_filter(float *input, Filter *filter, int x, int y) 
// {
// 	float sum = 0.0;
// 		for (int i = 0; i < FILTER_SIZE; i++)			// for accessing row of filter_mat
// 	{
// 		for (int j = 0; j < FILTER_SIZE; j++) 		// for accessing col of filter_mat
// 		{
// 			sum += input[x + i][y + j] * filter->weights[i][j];
// 		}
// 	}
// 	return sum + filter->bias; // Add bias
// }

// /*	Function: conv2d - convolution 2d of an single channels
// 	Input:
// 	Output:
// */
// void conv2d_1ch(float *input_mat, float *output_mat, Filter *filter, int input_mat_size) 
// {
// 	for (int i = 0; i < AFTER_KERNEL(input_mat_size); i++)
// 	{
// 		for (int j = 0; j < AFTER_KERNEL(input_mat_size); j++)
// 		{
// 			output[i][j] = apply_filter(input_mat, filter, i, j);
// 		}
// 	}
// }

/*
	i, j
		m, n
			p, q
	
*/
void conv2d(float *in_mat, float *out_mat, Filter *filter, int in_dim, int in_ch, int out_ch)
{
	int out_dim = AFTER_KERNEL(in_dim);
	// channels and kernels are bull$h!#
	for (int i_out_ch = 0; i_out_ch < out_ch; i_out_ch++)	// iterate though output channels
	{
		// START: iterate though out_mat to store value
		for (int m = 0; m < out_dim; m++)	// m for row
		{
			for (int n = 0; n < out_dim; n++)	// n for col
			{
				float sum = 0.0;
				for (int i_in_ch = 0; i_in_ch < in_ch; i_in_ch++)	// iterate though input channels
				{
					// START: iterate though Kernel

					for (int p = 0; p < FILTER_SIZE; p++)			// for accessing row of filter
					{
						for (int q = 0; q < FILTER_SIZE; q++) 		// for accessing col of filter
						{
							sum += (*(in_mat + (i_in_ch * FILTER_SIZE * FILTER_SIZE) + ((m + p) * FILTER_SIZE) + (n + q))) * filter[i_out_ch].weights[i_in_ch][p][q];
						}
					}
					// END: iterate though Kernel
				}
				*(out_mat + (i_out_ch * out_dim * out_dim) + (m * out_dim) +n) += sum + filter[i_out_ch].bias;
			}
		}
		// END: iterate though out_mat to store value
	}
}

/*	Function: relu
*/
void relu(float *mat, int mat_size, int num_ch) 
{
	for (int i_ch = 0; i_ch < num_ch; i_ch++)
	{
		for (int i = 0; i < mat_size; i++) 
		{
			for (int j = 0; j < mat_size; j++) 
			{
				float *ptr = (mat + (i_ch * mat_size * mat_size) + (i * mat_size) + j);
				if (*ptr < 0) 
				{
					*ptr = 0;
				}
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
// void fullyconnected(float input_mat[NUM_20CH][28][28], float output_mat[NUM_20CH][28][28], float *weight, float *bias, int input_mat_size, int output_mat_size)
// {
// 	// need to impliment matmul, but too reiji so ...
// 	// loop though the output mat to save cal result
// 	for (int i = 0; i < output_mat_size; i++)			// i for output_mat index
// 	{
// 		output_mat[i] = 0;
// 		// loop though the input mat to cal
// 		for (int j = 0; j < input_mat_size; j ++)		// j for input_mat index
// 		{
// 			output_mat[i] += input_mat[j] * weight[i][j];
// 		}
// 		output_mat[i] += bias[i]; 						// adding bias
// 	}
// }

/*
*/
void maxpooling(float input_mat[NUM_20CH][28][28], float output_mat[NUM_20CH][28][28], int input_mat_size, int pooling_size, int num_ch)
{
	 // Calculate the size of the output matrix
	int output_mat_size = (int)(input_mat_size / pooling_size);
	for (int i_ch = 0; i_ch < num_ch; i_ch++)
	{
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
						if (input_mat[i_ch][row_idx][col_idx] > max_value) {
							// max_value = input_mat[input_idx];
							max_value = input_mat[i_ch][row_idx][col_idx];

						}
					}
				}

				// Assign the maximum value to the output matrix
				// int output_idx = i * output_mat_size + j;
				// output_mat[output_idx] = max_value;
				output_mat[i_ch][i][j] = max_value;
			}
		}
	}
}

/*
*/
void printMatrix(float *mat, int num_ch, int mat_size, int external_arg)
{
	for (int i_ch = 0; i_ch < num_ch; i_ch++)
	{
		// External argument for printing extra info - mainly for debuging purpose
		if (external_arg == printKernelIdx)
		{
			printf("Kernel %d\n", i_ch);
		}
		else if (external_arg == printChannelIdx)
		{
			printf("Channel %d\n", i_ch);
		}
		else if (external_arg == printMatIdx)
		{
			printf("Matrix %d\n", i_ch);
		}

		// Printing loop
		for (int i = 0; i < mat_size; i++)
		{
			for (int j = 0; j < mat_size; j++)
			{
				printf("%f  ", *(mat + (i_ch * mat_size * mat_size) + (i * mat_size) + j));
			}
			printf("\n");
		}
		printf("\n");
	}
}

/*
*/
void load_weights(float weights[FILTER_SIZE][FILTER_SIZE], const float temp_weight[FILTER_SIZE][FILTER_SIZE])
{
	volatile uint8_t i, j;
	for (i = 0; i < FILTER_SIZE; i++) 
	{
	    for (j = 0; j < FILTER_SIZE; j++) 
	    {
	        weights[i][j] = temp_weight[i][j];
	    }
	}
}

void load_bias(Filter *filter, const float *temp_bias, int num_ch)
{
	volatile uint8_t i_ch;
	for (i_ch = 0; i_ch < num_ch; i_ch++)
	{
		filter[i_ch].bias = temp_bias[i_ch];
	}
}

/*
*/
void printFilterBias(Filter *filter, int num_ch)
{
	volatile uint8_t i_ch;
	for (i_ch = 0; i_ch < num_ch; i_ch++)
	{
		printf("%f ", filter[i_ch].bias);
	}
	printf("\n\n");
}

/*
*/
void printFilterWeight(float *mat, int num_ch_out, int num_ch_in, int mat_size)
{
	volatile uint8_t i_ch_out;
	for (i_ch_out = 0; i_ch_out < num_ch_out; i_ch_out++)
	{
		printf("Kernel %d\n", i_ch_out);
		printMatrix((float *)(mat + i_ch_out * num_ch_in * mat_size * mat_size));
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