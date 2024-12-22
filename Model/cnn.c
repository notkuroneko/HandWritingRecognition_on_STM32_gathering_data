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
#include <stdint.h>
#include <math.h>

#define __FLT_MAX__ 	(float)340282300000000000000000000000000000000

// #define LOL lol
#define INPUT_SIZE 		28
#define FILTER_SIZE 	3
#define AFTER_KERNEL(N) (N - FILTER_SIZE + 1)	// P = 0; S = 1
#define OUTPUT_SIZE 	10		// Number of output class, in this case 10 (0 to 9)

// some tinkering
#define IN_01CH 		1
#define IN_05CH 		5
#define IN_10CH 		10
#define IN_15CH 		15
#define IN_20CH 		20

typedef struct {
	float weights[IN_20CH][FILTER_SIZE][FILTER_SIZE];
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
// variable and stuff
	// input matrix data
	float input_mat[28][28] = {0};
	
	// layer 1 stuff - l1~~~
	uint8_t l1_dim = 28;
	uint8_t l1_ch = 5;
	float l1_mat[l1_ch][l1_dim][l1_dim] = {0};	// mat to store calculated value
	Filter conv1_filter[l1_ch] = {
	    {	// ch00
	        .weights = {
	            { 0.0124,  0.1875,  0.1134},
	            { 0.3180, -0.1194,  0.1318},
	            { 0.1256,  0.3247, -0.3463}
	        },
	        .bias = -0.2676
	    },
	    {	// ch01
	        .weights = {
	            {-0.1902,  0.2928,  0.2543},
	            { 0.2972,  0.0502,  0.0244},
	            { 0.1715, -0.1924,  0.2963}
	        },
	        .bias = -0.0370
	    },
	    {	// ch 02
	        .weights = {
	            {-0.5824, -0.3782, -0.6659},
	            {-0.4372, -0.6331, -0.6271},
	            {-0.6455, -0.5357, -0.6060}
	        },
	        .bias = 0.0005
	    },
	    {	// ch03
	        .weights = {
	            { 0.3625, -0.1436,  0.1321},
	            { 0.1683,  0.0952, -0.2777},
	            {-0.2249,  0.1154, -0.2253}
	        },
	        .bias = 0.2774
	    },
	    {	// ch04
	        .weights = {
	            { 0.3358,  0.0061, -0.2654},
	            { 0.1674,  0.3248, -0.2838},
	            { 0.2682,  0.3277, -0.3537}
	        },
	        .bias = 0.4532
	    }
	};
	
	// layer 2 stuff - l1~~~
	uint8_t l2_ch = 10;
	uint8_t l2_dim = 26;
	float l2_mat[l2_ch][l2_dim][l2_dim] = {0};	// mat to store calculated value
	Filter conv2_filter[l2_ch] = {
	    {   // ch00
	        .weights = {
				{
			        { 6.9994e-03, -1.0749e-01,  1.1241e-01 },
			        { -1.2853e-01, -8.2408e-02,  9.9403e-02 },
			        { 1.0726e-01, -8.2765e-02, -1.1805e-01 }
			    },
			    {
			        { -1.3585e-01,  5.3992e-03,  2.2752e-02 },
			        { -7.2987e-03,  1.0242e-01,  1.3990e-01 },
			        { 1.0460e-01, -6.6065e-02,  7.3180e-02 }
			    },
			    {
			        { 1.4108e-01, -1.2346e-01,  4.6408e-02 },
			        { 1.5450e-02, -7.5476e-02, -1.2136e-01 },
			        { 1.5864e-02, -2.3009e-02,  1.0366e-01 }
			    },
			    {
			        { 5.8840e-02,  1.1917e-01,  3.1105e-02 },
			        { -1.2554e-01,  9.4659e-02, -4.5635e-02 },
			        { 1.3812e-02,  1.3762e-02,  1.0526e-01 }
			    },
			    {
			        { -4.8520e-02, -1.3673e-01,  9.9640e-02 },
			        { -4.5110e-02, -5.6077e-02,  7.9168e-02 },
			        { -1.0438e-02, -6.5131e-02,  3.4479e-02 }
			    }
	        },
	        .bias = 0.0311
	    },
	    {   // ch01
	        .weights = {
    			{
			        { -6.2516e-02,  9.1627e-02, -5.5696e-02 },
			        {  1.0243e-02, -1.4965e-01,  1.3800e-01 },
			        {  3.3322e-02,  7.6210e-02,  3.1266e-02 }
			    },
			    {
			        {  1.3996e-01,  6.1691e-02, -4.1792e-02 },
			        {  1.1464e-01, -9.6571e-02, -7.8730e-02 },
			        { -1.4478e-01, -5.5349e-02,  7.6194e-02 }
			    },
			    {
			        {  4.1911e-02, -9.2358e-02,  3.5479e-02 },
			        { -1.1269e-02,  9.0401e-02,  1.3409e-01 },
			        { -1.1533e-01, -4.6837e-03, -8.3088e-02 }
			    },
			    {
			        { -6.8124e-02, -6.8642e-02,  1.8891e-04 },
			        { -2.3423e-02, -1.4134e-02, -9.2147e-02 },
			        {  2.6332e-02,  1.0499e-01, -1.3878e-01 }
			    },
			    {
			        {  1.7664e-02,  2.8719e-02,  9.6447e-02 },
			        { -1.4195e-01,  2.4718e-02, -1.4120e-01 },
			        {  8.3310e-04,  1.7864e-02,  1.4146e-01 }
			    }
	        },
	        .bias = -0.0646
	    },
	    {   // ch02
	        .weights = {
	            {
			        {  1.3994e-01,  1.2660e-02, -1.3319e-01 },
			        {  6.6428e-02, -1.8509e-02, -6.7482e-02 },
			        {  7.7382e-02,  5.7410e-02,  5.4714e-02 }
			    },
			    {
			        { -5.6701e-02, -5.3037e-02, -1.3398e-01 },
			        {  1.1907e-01, -1.3445e-01, -1.4918e-02 },
			        { -1.2833e-02,  4.2487e-02, -1.0797e-02 }
			    },
			    {
			        {  1.5464e-01,  6.1714e-02,  2.0213e-01 },
			        {  2.0526e-01,  2.3318e-01, -5.7323e-02 },
			        {  1.3460e-01,  3.2547e-02,  1.1777e-01 }
			    },
			    {
			        { -1.8554e-02,  1.0771e-01, -1.0323e-01 },
			        { -1.0559e-01,  6.8255e-02, -3.1022e-02 },
			        {  3.9565e-02,  3.9962e-02,  2.4795e-02 }
			    },
			    {
			        {  2.4476e-04, -1.0110e-01, -2.7256e-02 },
			        { -2.6879e-03, -6.5805e-03, -5.6743e-03 },
			        { -1.0562e-01,  7.0542e-02, -2.0944e-03 }
			    }
	        },
	        .bias = 0.0503
	    },
	    {   // ch03
	        .weights = {
                {
			        { -6.5106e-02, -1.4890e-01, -2.4382e-02 },
			        { -6.7631e-02,  8.1379e-02, -8.2115e-02 },
			        {  4.2132e-03, -5.3786e-02, -1.5664e-01 }
			    },
			    {
			        { -4.8399e-02, -6.4995e-02, -8.5580e-02 },
			        {  6.5211e-02, -4.6275e-02, -1.4144e-01 },
			        { -4.9687e-03, -3.6003e-02, -1.2884e-01 }
			    },
			    {
			        {  3.5966e-01,  2.7275e-01,  4.0809e-01 },
			        {  1.5739e-01,  4.2356e-01,  3.4814e-01 },
			        {  1.7740e-01,  3.4385e-01,  2.5349e-01 }
			    },
			    {
			        { -3.5987e-02, -1.1540e-01, -3.8754e-02 },
			        { -8.4724e-02,  1.3516e-01, -4.0697e-02 },
			        {  2.8031e-02,  4.7956e-02, -8.1787e-02 }
			    },
			    {
			        { -9.8221e-02,  3.4153e-02, -1.8770e-01 },
			        { -1.4707e-01, -1.7612e-01, -1.3713e-01 },
			        { -1.8266e-01, -8.0488e-02, -5.1264e-02 }
			    }
	        },
	        .bias = 0.1601
	    },
	    {   // ch04
	        .weights = {
                {
			        {  9.9442e-02,  1.2562e-02, -1.5019e-01 },
			        {  7.1585e-02, -1.7437e-02, -2.7939e-02 },
			        { -9.6054e-02,  1.0464e-01,  1.0440e-01 }
			    },
			    {
			        { -4.4948e-02,  2.5588e-02,  9.7390e-02 },
			        { -9.9155e-02,  1.3481e-02,  5.7662e-03 },
			        { -5.0368e-02, -1.2909e-01,  2.8813e-02 }
			    },
			    {
			        {  2.1639e-01,  3.8027e-01,  2.1209e-01 },
			        {  3.4494e-01,  3.5115e-01,  2.6501e-01 },
			        {  3.7366e-01,  3.2540e-01,  2.3323e-01 }
			    },
			    {
			        {  6.2991e-02, -1.8020e-02,  1.2608e-01 },
			        {  7.0571e-02, -7.1504e-02, -3.6252e-02 },
			        { -6.0938e-02,  4.1955e-02, -9.8371e-02 }
			    },
			    {
			        {  2.7646e-02,  8.7350e-03, -1.3798e-01 },
			        { -1.9231e-01, -1.5626e-01,  6.6979e-02 },
			        { -1.2564e-02, -1.2956e-01, -1.3629e-01 }
			    }
	        },
	        .bias = 0.0339
	    },
	    {   // ch05
	        .weights = {
                {
			        {  1.3382e-01, -6.6101e-02, -1.1319e-01 },
			        {  1.9125e-01,  8.9179e-03, -7.5039e-02 },
			        {  1.7117e-01, -1.3839e-01,  1.2194e-02 }
			    },
			    {
			        {  9.0309e-02,  1.1531e-01,  1.0453e-01 },
			        {  1.4739e-01, -2.5521e-02, -1.2480e-01 },
			        {  1.5953e-01, -4.1523e-02, -1.5040e-01 }
			    },
			    {
			        { -2.5259e-01, -1.1847e-01, -2.5554e-02 },
			        { -1.2140e-01, -9.2689e-02,  8.8769e-02 },
			        {  9.9991e-02,  1.3761e-01,  2.4332e-01 }
			    },
			    {
			        { -1.1140e-01, -8.1607e-02,  6.8346e-02 },
			        {  5.7895e-02,  1.0981e-01,  1.4226e-01 },
			        {  7.5203e-02,  1.4228e-01, -1.2690e-02 }
			    },
			    {
			        {  7.3175e-02, -2.0817e-02,  2.2969e-01 },
			        {  2.7637e-01,  1.7079e-01, -4.0658e-02 },
			        {  2.5030e-01,  1.0877e-01, -4.9040e-03 }
			    }
	        },
	        .bias = 0.1120
	    },
	    {   // ch06
	        .weights = {
                {
			        { -4.9395e-02,  7.7535e-02, -3.0308e-02 },
			        {  5.3317e-02, -9.5195e-03,  8.6248e-02 },
			        { -9.0979e-02,  9.0059e-02,  1.4335e-01 }
			    },
			    {
			        { -5.9521e-02, -1.0584e-01,  8.1378e-02 },
			        {  3.7128e-02, -1.3285e-02,  9.6754e-02 },
			        {  1.1390e-01,  4.4429e-02,  1.0111e-01 }
			    },
			    {
			        {  6.4477e-03, -4.6143e-02, -9.9494e-02 },
			        { -2.7805e-02,  4.0339e-02,  9.9859e-02 },
			        { -1.0813e-01,  4.1479e-02,  6.2407e-02 }
			    },
			    {
			        { -3.8753e-02,  1.0396e-02, -5.3173e-02 },
			        {  1.7906e-02,  6.7314e-02,  1.5898e-01 },
			        { -6.2624e-02, -7.0675e-02,  1.8046e-01 }
			    },
			    {
			        {  9.5974e-02, -1.2103e-01,  2.3157e-02 },
			        { -1.2367e-01, -2.5929e-02,  1.3791e-01 },
			        { -2.3650e-02,  6.8873e-02,  1.7835e-02 }
			    }
	        },
	        .bias = 0.1015
	    },
	    {   // ch07
	        .weights = {
		        {
			        {-1.2015e-01, -3.6844e-02, -7.6767e-02},
			        { 7.8797e-02, -4.7797e-03,  9.8685e-02},
			        {-1.4712e-01,  9.5983e-02, -8.9038e-02}
		    	},
		        {
		        	{ 3.4584e-02, -1.6962e-03,  2.9747e-02},
			        {-4.0723e-02,  3.4913e-02,  9.6485e-02},
			        { 7.4472e-02, -1.4148e-01, -1.2437e-01}
		    	},
		        {
					{ 2.0658e-01,  1.5613e-01,  9.1583e-02},
					{ 1.5452e-01,  2.7130e-01,  3.1086e-01},
					{ 2.8304e-01,  2.4689e-01,  1.9325e-01}
		     	},
		        {
		        	{ 1.2642e-01, -1.1946e-02,  5.8047e-02},
		         	{-6.9046e-02, -1.1501e-01, -7.5056e-02},
		         	{-1.3689e-02, -6.9413e-02,  1.3454e-01}
		        },
		        {
		        	{-5.3500e-02, -1.3793e-01, -2.8221e-02},
		         	{-4.9056e-02, -3.0885e-02, -1.3797e-01},
		         	{-1.4480e-01,  7.8703e-03, -6.0094e-02}
		        }
	        },
	        .bias = -0.0644
	    },
	    {   // ch08
	        .weights = {
				{
					{-1.0990e-02, -2.7940e-02, -3.7894e-02},
					{-7.7178e-02, -8.7366e-02,  1.3665e-01},
					{-3.3196e-02, -3.5389e-02,  4.5041e-02}
				},
				{
					{-1.8844e-02,  6.0720e-02, -3.0335e-03},
					{-7.4671e-02,  2.7046e-02,  1.3539e-02},
					{ 8.7976e-02,  2.6804e-02, -1.0660e-01}
				},
				{
					{-2.1324e-02, -2.7471e-03,  9.7499e-02},
					{ 1.5060e-01,  1.8537e-01,  2.8545e-01},
					{-3.3071e-02,  1.9289e-01,  2.6050e-01}
				},
				{
					{-4.7628e-02, -1.9439e-02,  2.4740e-02},
					{-1.0815e-01, -3.9849e-04,  2.3234e-02},
					{ 7.7282e-03, -6.9537e-02,  7.9489e-02}
				},
				{
					{-2.6737e-02,  5.1174e-02, -6.9467e-02},
					{ 6.3165e-02, -4.5820e-02, -1.3789e-01},
					{ 1.1573e-02, -3.5946e-04, -1.0541e-01}
				}
	        },
	        .bias = -0.0354
	    },
	    {   // ch09
	        .weights = {
				{
					{ 3.2992e-02, -1.2790e-01, -1.0268e-01},
					{ 1.3143e-02, -1.4100e-01, -7.6164e-02},
					{-5.8229e-02, -1.3957e-01, -8.7025e-02}
				},
				{
					{-1.4264e-02, -1.4167e-01, -1.2305e-01},
					{-1.3940e-01, -1.0484e-01,  1.3709e-02},
					{ 8.0303e-02,  1.2909e-01, -1.0265e-01}
				},
				{
					{ 7.0418e-02, -4.0901e-02, -1.0520e-01},
					{ 9.5288e-02, -9.0211e-02, -3.5670e-02},
					{-8.8031e-02, -9.8073e-02,  1.3701e-01}
				},
				{
					{-7.9061e-02, -1.0308e-01,  2.7901e-02},
					{ 6.0610e-02, -7.5722e-02, -3.8225e-03},
					{ 4.1306e-02, -9.9969e-02, -1.1973e-01}
				},
				{
					{ 3.7578e-02, -1.4554e-01, -1.6944e-02},
					{ 3.1978e-02,  5.6472e-02,  5.7986e-02},
					{-1.6333e-02, -2.7773e-02, -3.8826e-02}
				}
	        },
	        .bias = -0.0236
	    }
	};

	// layer 3 stuff
	// uint8_t l3_ch = 10;
	// uint8_t l3_dim = 26;
	// float l3_mat[l3_ch][l3_dim][l3_dim] = {0};	// mat to store calculated value
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

	// // layer 4 stuff
	// uint8_t l4_ch = 20

	// fullyconnected weight and bias
	// float fc_weight[][] = {0};
	// float fc_bias[] = {0};

// Calculation
	// layer 1
	for (int i = 0; i < l1_ch; i++)		// iterate though layer 1 channels
	{
		conv2d(&input_mat[0][0], layer1ch00_mat, filter_conv1, l1_dim);
	}

	// layer 2

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
	for (int )
		for (int i = 0; i < FILTER_SIZE; i++)			// for accessing row of filter_mat
		{
			for (int j = 0; j < FILTER_SIZE; j++) 		// for accessing col of filter_mat
			{
				sum += input[x + i][y + j] * filter->weights[i][j];
			}
		}
	return sum + filter->bias; // Add bias
}

/*	Function: conv2d - convolution 2d of an single channels
	Input:
	Output:
*/
void conv2d(float *input_mat, float *output_mat, Filter *filter, int input_mat_size) 
{
	for (int i = 0; i < AFTER_KERNEL(input_mat_size); i++)
	{
		for (int j = 0; j < AFTER_KERNEL(input_mat_size); j++)
		{
			output[i][j] = apply_filter(input, filter, i, j);
		}
	}
}

/*
	i, j
		m, n
			p, q
	
*/
// void conv2d(float *in_mat, float *out_mat, Filter *filter, int in_dim, int in_ch, int out_ch)
// {
// 	// channels and kernels are bull$h!#
// 	for (int i = 0; i < out_ch; i++)	// iterate though output channels matrix using i
// 	{
// 		for (int j = 0; j < in_ch; j++)	// iterate though input channels matrix using j
// 		{
// 			// START: iterate though out_mat to store value
// 			for (int m = 0; m < AFTER_KERNEL(in_dim); m++)	// m for row
// 			{
// 				for (int n = 0; n < AFTER_KERNEL(in_dim); n++)	// n for col
// 				{
// 					// out_mat[i][m][n] = apply_filter(in_mat, filter, m, n);
// 					// START: iterate though Kernel
// 					float sum = 0.0;
// 					for (int p = 0; p < FILTER_SIZE; p++)			// for accessing row of filter
// 					{
// 						for (int q = 0; q < FILTER_SIZE; q++) 		// for accessing col of filter
// 						{
// 							sum += in_mat[j][m + p][n + q] * filter[i]->weights[p][q];
// 						}
// 					}
// 					// END: iterate though Kernel
// 					out_mat[i][m][n] += sum;
// 				}
// 			}
// 			// END: iterate though out_mat to store value
// 		}

// 	}
// }

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