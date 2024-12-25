/*
	File: cnn.c
	Author: Tran Ba Thanh
	Created: 21/12/2024
	Last update: 21/12/2024
	Purpose: to impliment trained weight from python in C

	Todo:
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
#include "conv3_params.h"
#include "conv4_params.h"
#include "lin1_params.h"
#include "lin2_params.h"

#define DEBUG 0

// size = 28*28*1*4 = 3136 bytes
const float input_mat_temp[INPUT_SIZE * INPUT_SIZE] = {
	0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 1.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 1.0, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 1.0, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 1.0, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 1.0, 1.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.9, 1.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 1.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 1.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 1.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 1.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 1.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 1.0, 1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 1.0, 1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 1.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 1.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,


};

int main()
{
// variable and stuff
	
	// layer 1 stuff - l1~~~
	const int l1_ch = NUM_05CH;
	const int l1_dim = 26;
	float l1_mat[l1_ch * l1_dim * l1_dim];	// mat to store calculated value
	int l1_mat_bytesize = sizeof(float)*l1_ch*l1_dim*l1_dim;
	memset(l1_mat, 0, l1_mat_bytesize);

	// layer 2 stuff - l1~~~
	const int l2_ch = NUM_10CH;
	const int l2_dim = 24;
	float l2_mat[l2_ch * l2_dim * l2_dim];	// mat to store conv2 calculated value
	int l2_mat_bytesize = sizeof(float)*l2_ch*l2_dim*l2_dim;
	memset(l2_mat, 0, l2_mat_bytesize);
	const int l2_pool_mat_dim = 12; 		// l2_pool_mat_dim = l2_dim / 2
	float l2_pool_mat[l2_ch * l2_pool_mat_dim * l2_pool_mat_dim];			// mat to store maxpooling calculated value
	int l2_pool_mat_bs = sizeof(float)*l2_ch*l2_pool_mat_dim*l2_pool_mat_dim;
	memset(l2_pool_mat, 0, l2_pool_mat_bs);

	// layer 3 stuff
	const int l3_ch = NUM_12CH;
	const int l3_dim = 10;
	float l3_mat[l3_ch * l3_dim * l3_dim];	// mat to store calculated value
	int l3_mat_bs = sizeof(float) * l3_ch * l3_dim * l3_dim;
	memset(l3_mat, 0, l3_mat_bs);

	// layer 4 stuff
	const int l4_ch = NUM_15CH;
	const int l4_dim = 8;
	float l4_mat[l4_ch * l4_dim * l4_dim];	// mat to store conv2 calculated value
	int l4_mat_bytesize = sizeof(float) * l4_ch * l4_dim * l4_dim;
	memset(l4_mat, 0, l4_mat_bytesize);
	const int l4_pool_mat_dim = 4; 		// l2_pool_mat_dim = l2_dim / 2
	float l4_pool_mat[l4_ch * l4_pool_mat_dim * l4_pool_mat_dim];			// mat to store maxpooling calculated value
	int l4_pool_mat_bs = sizeof(float) * l4_ch * l4_pool_mat_dim * l4_pool_mat_dim;
	memset(l4_pool_mat, 0, l4_pool_mat_bs);

	// fullyconnected
	float fc1_mat[FC02NODENUM];	// mat to store fullyconnected 1
	int fc1_mat_bytesize = sizeof(float) * FC02NODENUM;
	memset(fc1_mat, 0, fc1_mat_bytesize);

	float fc2_mat[OUTPUT_SIZE];	// mat to sotre fullyconnected 2
	int fc2_mat_bytesize = sizeof(float) * OUTPUT_SIZE;
	memset(fc2_mat, 0, fc2_mat_bytesize);

	float softmax_result_mat[OUTPUT_SIZE];
	int softmax_result_mat_size = sizeof(float) * OUTPUT_SIZE;
	memset(softmax_result_mat, 0, softmax_result_mat_size);

// Calculation
	// layer 1
	conv2d((float *)input_mat_temp, (float *)l1_mat, (float *)conv1_filter_weight, (float *)conv1_filter_bias, INPUT_SIZE, NUM_01CH, NUM_05CH);
	relu((float *)l1_mat, AFTER_CONV1_DIM, NUM_05CH);

	// layer 2
	conv2d((float *)l1_mat, (float *)l2_mat, (float *)conv2_filter_weight, (float *)conv2_filter_bias, AFTER_CONV1_DIM, NUM_05CH, NUM_10CH);
	relu((float *)l2_mat, AFTER_CONV2_DIM, NUM_10CH);
	maxpooling2x2((float *)l2_mat, (float *)l2_pool_mat, AFTER_CONV2_DIM, NUM_10CH);

	// layer 3
	conv2d((float *)l2_pool_mat, (float *)l3_mat, (float *)conv3_filter_weight, (float *)conv3_filter_bias, AFTER_MAXP1_DIM, NUM_10CH, NUM_12CH);
	relu((float *)l3_mat, AFTER_CONV3_DIM, NUM_12CH);	

	// layer 4
	conv2d((float *)l3_mat, (float *)l4_mat, (float *)conv4_filter_weight, (float *)conv4_filter_bias, AFTER_CONV3_DIM, NUM_12CH, NUM_15CH);
	relu((float *)l4_mat, AFTER_CONV3_DIM, NUM_15CH);
	maxpooling2x2((float *)l4_mat, (float *)l4_pool_mat, AFTER_CONV4_DIM, NUM_10CH);

	fullyconnected((float *)l4_pool_mat, (float *)fc1_mat, (float *)lin1_weight, (float *)lin1_bias, FC01NODENUM, FC02NODENUM);	

	fullyconnected((float *)fc1_mat, (float *)fc2_mat, (float *)lin2_weight, (float *)lin2_bias, FC02NODENUM, OUTPUT_SIZE);	

	softmax((float *)fc2_mat, (float *)softmax_result_mat, OUTPUT_SIZE);

	printf("\n=======================================================================\n");
	printMat1D((float *)softmax_result_mat, NUM_10CH);
	printf("=======================================================================\n");
	printPredict((float *)softmax_result_mat);

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
void conv2d(float *in_mat, float *out_mat, float *filter_weight, float *filter_bias, int in_dim, int in_ch, int out_ch)
{
	int out_dim = AFTER_KERNEL(in_dim);	// calculate out_mat dimentions

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
							float *in_mat_idx = in_mat 
															+ (i_in_ch * FILTER_SIZE * FILTER_SIZE) 
															+ ((m + p) * FILTER_SIZE) 
															+ (n + q);
							float *filter_weight_idx = filter_weight 
																+ ((i_out_ch * in_ch + i_in_ch) * FILTER_SIZE * FILTER_SIZE)
																+ p * FILTER_SIZE
																+ q;
							// sum += in_mat[i_ch_out][m+p][n+q] * filter_weight[i_out_ch][i_in_ch][p][q]
							sum += (*(in_mat_idx)) * (*(filter_weight_idx));
						}
					}
					// END: iterate though Kernel
				}
				*(out_mat + (i_out_ch * out_dim * out_dim) + (m * out_dim) +n) += sum + (*(filter_bias + i_out_ch));
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
void softmax(float *input_mat, float *out_mat, int mat_size)
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
		out_mat[i] = exp(input_mat[i]) / total_exp;
	}
}

/*	Function: fullyconnected
	Input:
	Output:
*/
void fullyconnected(float *in_mat, float *out_mat, float *weight, float *bias, int in_mat_size, int out_mat_size)
{
	// need to impliment matmul, but too reiji so ...
	// loop though the output mat to save cal result
	for (int i_out = 0; i_out < out_mat_size; i_out++)			// i_out for out_mat index
	{
		out_mat[i_out] = 0;
		// loop though the input mat to cal
		for (int i_in = 0; i_in < in_mat_size; i_in ++)		// j for input_mat index
		{
			out_mat[i_out] += in_mat[i_in] * weight[i_out * in_mat_size + i_in];
		}
		out_mat[i_out] += bias[i_out]; 						// adding bias
	}
}

/*
*/
void maxpooling2x2(float *in_mat, float *out_mat, uint8_t in_mat_size, uint8_t num_ch)
{
	 // Calculate the size of the output matrix
	uint8_t out_mat_size = (int)(in_mat_size / 2);

	for (int i_ch = 0; i_ch < num_ch; i_ch++)
	{
		// Iterate through the output matrix
		for (int i = 0; i < out_mat_size; i++) 
		{
			for (int j = 0; j < out_mat_size; j++)
			{
				float max_value = -__FLT_MAX__; // Initialize to the smallest float value
				
				// Iterate through the pooling window
				for (int m = 0; m < POOLING_SIZE; m++) 
				{
					for (int n = 0; n < POOLING_SIZE; n++) 
					{
						// Calculate the index of the current input element
						int row_idx = i * POOLING_SIZE + m;
						int col_idx = j * POOLING_SIZE + n;
						int input_idx = row_idx * in_mat_size + col_idx;

						// Update the maximum value
						if (in_mat[input_idx] > max_value) {
							max_value = in_mat[input_idx];
							// max_value = input_mat[i_ch][row_idx][col_idx];
						}
					}
				}
				// Assign the maximum value to the output matrix
				// output_mat[i_ch][i][j] = max_value;
				int output_idx = i * out_mat_size + j;
				out_mat[output_idx] = max_value;
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
void printFilterBias(float *bias_mat, int num_ch)
{
	for (int i_ch = 0; i_ch < num_ch; i_ch++)
	{
		printf("%f ", bias_mat[i_ch]);
	}
	printf("\n\n");
}

void printMat1D(float *mat, int num_ch)
{
	printFilterBias((float *)mat, num_ch);
}

/*
*/
void printFilterWeight(float *mat, int num_ch_out, int num_ch_in, int mat_size)
{
	for (int i_ch_out = 0; i_ch_out < num_ch_out; i_ch_out++)
	{
		printf("Kernel %d\n", i_ch_out);
		printMatrix((float *)(mat + i_ch_out * num_ch_in * mat_size * mat_size), num_ch_in, mat_size, printChannelIdx);
	}
}

void printPredict(float *mat)
{
	float max_predict = mat[0];
	int max_idx = 0;
	for (int i = 1; i < OUTPUT_SIZE; i++)
	{
		if (max_predict < mat[i])
		{
			max_predict = mat[i];
			max_idx = i;
		}
	}
	printf("Predict %d with chance of %f\n", max_idx, max_predict);
}

// void matmul_a_b()
// void matmul_a_bt()
// void matmul_at_b()
/*
	Reference:
		https://github.com/usamahz/cnn.git
		https://youtu.be/jDe5BAsT2-Y?si=i9AY3u8cukuy75CI
		
*/