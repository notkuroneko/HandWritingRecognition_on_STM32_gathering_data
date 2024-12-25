#ifndef CNN_H
#define CNN_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "constants.h"
#include "conv1_params.h"
#include "conv2_params.h"

// ----------------Function prototype----------------
// Convolution functions
// void read_input_data(const char *filename, float input[INPUT_SIZE][INPUT_SIZE]);
// float apply_filter(float *input, Filter *filter, int x, int y) ;
// void conv2d_1ch(float *input_mat, float *output_mat, Filter *filter, int input_mat_size);
void conv2d(float *in_mat, float *out_mat, Filter *filter, int in_dim, int in_ch, int out_ch);
void relu(float *mat, int mat_size, int num_ch);
void softmax(float *input_mat, float *output_mat, int mat_size);
// void fullyconnected(float input_mat[NUM_20CH][28][28], float output_mat[NUM_20CH][28][28], float *weight, float *bias, int input_mat_size, int output_mat_size);
void maxpooling(float input_mat[NUM_20CH][28][28], float output_mat[NUM_20CH][28][28], int input_mat_size, int pooling_size, int num_ch);
void load_bias(Filter *filter, const float *temp_bias, int num_ch);
void load_weights(float weights[FILTER_SIZE][FILTER_SIZE], const float temp_weight[FILTER_SIZE][FILTER_SIZE]);
// Convollution layer funciton
// void;

// Ultility functions
void printMatrix(float *mat, int num_ch, int mat_size, int external_arg);
void printFilterBias(Filter *filter, int num_ch);
void printFilterWeight(float *mat, int num_ch_out, int num_ch_in, int mat_size);

#endif // CNN_H