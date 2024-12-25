#ifndef CONSTANTS_H
#define CONSTANTS_H

#define INPUT_SIZE 		28
#define FILTER_SIZE 	3
#define AFTER_KERNEL(N) (N - FILTER_SIZE + 1)	// P = 0; S = 1
#define OUTPUT_SIZE 	10		// Number of output class, in this case 10 (0 to 9)

// some tinkering
#define NUM_01CH 		1
#define NUM_05CH 		5
#define NUM_10CH 		10
#define NUM_12CH		12
#define NUM_15CH 		15
#define NUM_20CH 		20

// CNN kernel numbrer
#define CONV1_K_NUM		NUM_05CH
#define CONV2_K_NUM		NUM_10CH
#define CONV3_K_NUM		NUM_12CH
#define CONV4_K_NUM		NUM_15CH

enum ExternalArg
{
	// nonExternalArg,
	printKernelIdx,
	printChannelIdx,
	printMatIdx,
};

#endif // CONSTANTS_H