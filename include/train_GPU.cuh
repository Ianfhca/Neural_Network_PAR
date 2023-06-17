#ifndef __GPU_TRAIN_H
#define __GPU_TRAIN_H

// #include <cuda.h>
// #include <cuda_runtime.h>

#include "ds.h"
#include "matrix.cuh"
#include "nn.cuh"
#include "nn_aux.h"

__device__ void matrix_mul_add_GPU(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols, double *d){
    int i, col, row;
    double sum;

    for (row = 0; row < a_rows; row++) {
        for (col = 0; col < b_cols; col++) {
            sum = 0.0;
            for (i = 0; i < a_cols; i++) {
                sum += a[row * a_cols + i] * b[i * b_cols + col];
            }
            c[row * b_cols + col] = d[row * b_cols + col] + sum;        
        }
    }
}

// __device__ void forward_pass_GPU(nn_t *nn, double *input, double **A, double **Z, int tid) {
//     int i;
//     // Z[0] = 2121212;
//     for (i = 0; i < nn->layers_size[0]; i++) {
//         A[0][i] = input[i];
//     }
//     // if ((i = tid % nn->n_layers) != 0) {
//     //     cuda_matrix_mul_add(Z[i], nn->WH[i - 1], A[i - 1], nn->layers_size[i], nn->layers_size[i - 1], nn->layers_size[i - 1], 1, nn->BH[i - 1]);
//     //     // matrix_func(A[i], Z[i], nn->layers_size[i], 1, nn->activation_ptr[i - 1]);
//     //     // matrix_func(Z[i], Z[i], nn->layers_size[i], 1, nn->dactivation_ptr[i - 1]);
//     //     printf("TID: %d\n", tid);
//     // }
//     printf("layers_size = %d\n", nn->layers_size[0]);
// }

#endif