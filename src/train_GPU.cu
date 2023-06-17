// #ifndef MY_DEVICE_KERNELS_CU
// #define MY_DEVICE_KERNELS_CU

// #ifdef GPU
// #include "train.h"

// __device__ void forward_pass_GPU(nn_t *nn, double *input, double *A, double *Z) {
    // int i;

    // for (i = 0; i < nn->layers_size[0]; i++) {
    //     A[0][i] = input[i];
    // }

    // for (i = 1; i < nn->n_layers; i++) {
    //     matrix_mul_add(Z[i], nn->WH[i - 1], A[i - 1], nn->layers_size[i], nn->layers_size[i - 1], nn->layers_size[i - 1], 1, nn->BH[i - 1]);
    //     matrix_func(A[i], Z[i], nn->layers_size[i], 1, nn->activation_ptr[i - 1]);
    //     matrix_func(Z[i], Z[i], nn->layers_size[i], 1, nn->dactivation_ptr[i - 1]);
    // }

    // int col = blockIdx.x * blockDim.x + threadIdx.x;
    // int row = blockIdx.y * blockDim.y + threadIdx.y;

    // int i;
    // float sum;

    // if (col < b_cols && row < a_rows) {
    //     sum = 0;
    //     // Recorrer las filas para cada elemento de la matriz
    //     for (i = 0; i < b_rows; i++) {
    //         sum += d_WH[row * a_cols + i] * d_A[i * b_cols + col];
    //     }
    //     d_Z[row * b_cols + col] = d_BH[row * b_cols + col] + sum;
    // }
//     printf("HOLA\n");
// }
// #endif
// #endif