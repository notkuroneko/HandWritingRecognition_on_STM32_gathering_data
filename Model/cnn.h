#ifdef CNN_H
#define CNN_H

#include <stdint.h>

// #define LOL lol
extern const uint8_t INPUT_SIZE = 28;
extern const uint8_t FILTER_SIZE = 3;
#define AFTER_KERNEL(N) (uint8_t)(N - FILTER_SIZE + 1)	// P = 0; S = 1
extern const uint8_t OUTPUT_SIZE = 10;		// Number of output class, in this case 10 (0 to 9)

// Number of channel(s)
extern const uint8_t NUM_01CH = 1;
extern const uint8_t NUM_05CH = 5;
extern const uint8_t NUM_10CH = 10;
extern const uint8_t NUM_15CH = 15;
extern const uint8_t NUM_20CH = 20;

typedef struct {
	float weights[NUM_20CH][FILTER_SIZE][FILTER_SIZE];
	float bias;
} Filter;

// ----------------Function prototype----------------
// Convolution functions
// void read_input_data(const char *filename, float input[INPUT_SIZE][INPUT_SIZE]);
// float apply_filter(float *input, Filter *filter, int x, int y) ;
// void conv2d_1ch(float *input_mat, float *output_mat, Filter *filter, int input_mat_size);
void conv2d(float in_mat[NUM_20CH][28][28], float out_mat[NUM_20CH][28][28], Filter *filter, int in_dim, int in_ch, int out_ch);
void relu(float mat[NUM_20CH][28][28], int mat_size, int num_ch);
void softmax(float *input_mat, float *output_mat, int mat_size);
// void fullyconnected(float input_mat[NUM_20CH][28][28], float output_mat[NUM_20CH][28][28], float *weight, float *bias, int input_mat_size, int output_mat_size);
void maxpooling(float input_mat[NUM_20CH][28][28], float output_mat[NUM_20CH][28][28], int input_mat_size, int pooling_size, int num_ch);

// Ultility functions
void printMatrix(float mat[NUM_20CH][28][28], int num_ch, int mat_size);

#endif // CNN_H