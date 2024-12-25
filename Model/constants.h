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
#define NUM_15CH 		15
#define NUM_20CH 		20

#endif // CONSTANTS_H