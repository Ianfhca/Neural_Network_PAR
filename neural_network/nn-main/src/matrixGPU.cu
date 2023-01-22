#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "matrixGPU.cuh"
#include "nn_aux.h"
#include "globals.h"

#include <cuda_runtime.h>

#ifdef TIMING
    #include <time.h>
    #include "utils.h"
#endif

dim3 dimBlock, dimGrid;

__global__ void mul_add(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols, double *d) {
    
    
}


// __global__ void cuda_mat_sum(float *A, float *B, float *C, int a, int b) {
//     int i = blockIdx.y*blockDim.y + threadIdx.y;
//     int j = blockIdx.x*blockDim.x + threadIdx.x;
//     if (i < b && j < a) {
//         C[i*a + j] = A[i*a + j] + B[i*a + j];
//     }
// }

__global__ void cuda_mat_sum(double *C, double *A, double *B, int rows, int cols) {
    int i = blockIdx.x *blockDim.x + threadIdx.x;
    if (i < rows*cols) {
        C[i] = A[i] + B[i];
    }
}

__global__ void cuda_mat_sub(double *C, double *A, float *B, int rows, int cols) {
    int i = blockIdx.x *blockDim.x + threadIdx.x;
    if (i < rows*cols) {
        C[i] = A[i] - B[i];
    }
}

__global__ void cuda_mat_cnt(double *M, int rows, int cols, double cnt) {
    int i = blockIdx.x *blockDim.x + threadIdx.x;
    if (i < rows*cols) {
        M[i] = cnt;
    }
}

__global__ void cuda_mul_dot(double *C, double *A, double *B, int rows, int cols) {
    int i = blockIdx.x *blockDim.x + threadIdx.x;
    if (i < rows*cols) {
        C[i] = A[i] * B[i];
    }
}

__global__ void cuda_matrix_transpose(double *M, double *AUX, int rows, int cols) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int row = id / cols;
    int col = id % cols;
    if (id < rows*cols) {
        M[rows*col + row] = AUX[cols*row + col];
    }
}

__global__ void cuda_matrix_mul(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols) {
    // int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // int row = blockIdx.y * blockDim.y + threadIdx.y;
    // int col = blockIdx.x * blockDim.x + threadIdx.x;
    // float value = 0;
    // if (row < a_row && col < b_cols) {
    //     value += a[row * a_cols + i] * b[i * b_cols + col];
    // }
    // c[row * b_cols + col] = value;
}


__global__ void cuda_matrix_mul_add(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols, double *d) {

}

double **GPU_alloc_matrix_2v(int n_layers, int *size, int *size_prev, double (*init_weight_ptr)(void)) {

    double **m, **aux, **M;
    int i, j;
    cudaError_t err = cudaGetLastError();

    if ((m = (double**)malloc(n_layers * sizeof(double*))) == NULL) {
        return(NULL);
    }

    if ((aux = (double**)malloc(n_layers * sizeof(double*))) == NULL) {
        return(NULL);
    }

    if((err = cudaMalloc(&M, n_layers * sizeof(double*))) != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return NULL;
    }

    for (i=0; i < n_layers; i++) {
        if ((m[i] = (double*)malloc(size[i] * size_prev[i] * sizeof(double))) == NULL) {
            // matrix_free_2D(m, n_layers);
            return(NULL);
        }
        if ((aux[i] = (double*)malloc(size[i] * size_prev[i] * sizeof(double))) == NULL) {
            // matrix_free_2D(m, n_layers);
            return(NULL);
        }
        if((err = cudaMalloc(&m[i], size[i] * size_prev[i] * sizeof(double))) != cudaSuccess){
            GPU_matrix_free_2D(M, n_layers);
            return NULL;
        }
    }

    for (i = 0; i < n_layers; i++){
        for (j =0; j < size[i] * size_prev[i]; j++){
            aux[i][j] = init_weight_ptr();
        }
        cudaMemcpy(m[i], aux[i], size[i] * size_prev[i] * sizeof(double), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(M, m, n_layers * sizeof(double*), cudaMemcpyHostToDevice);

    return(M);
}

double **GPU_alloc_matrix_1v(int n_layers, int *size, double (*init_weight_ptr)(void)) {

    double **m, **aux, **M;
    int i, j;
    cudaError_t err = cudaGetLastError();

    if ((m = (double**)malloc(n_layers * sizeof(double*))) == NULL) {
        return(NULL);
    }

    if ((aux = (double**)malloc(n_layers * sizeof(double*))) == NULL) {
        return(NULL);
    }

    if((err = cudaMalloc(&M, n_layers * sizeof(double*))) != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return NULL;
    }

    for (i=0; i < n_layers; i++) {
        if ((m[i] = (double*)malloc(size[i] * sizeof(double))) == NULL) {
            // matrix_free_2D(m, n_layers);
            return(NULL);
        }
        if ((aux[i] = (double*)malloc(size[i] * sizeof(double))) == NULL) {
            // matrix_free_2D(m, n_layers);
            return(NULL);
        }
        if((err = cudaMalloc(&m[i], size[i] * sizeof(double))) != cudaSuccess){
            GPU_matrix_free_2D(M, n_layers);
            return NULL;
        }
    }

    for (i = 0; i < n_layers; i++){
        for (j =0; j < size[i]; j++){
            aux[i][j] = init_weight_ptr();
        }
        cudaMemcpy(m[i], aux[i], size[i] * sizeof(double), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(M, m, n_layers * sizeof(double*), cudaMemcpyHostToDevice);

    return(M);
}

double *GPU_alloc_array(int length){

    double *v;
    int i;

    if ((v = (double*)malloc(length* sizeof(double))) == NULL) {
        return(NULL);
    }

    for (i = 0; i < length; i++){
        v[i] = 0.0;
    }
    
    return(v);
}


double *GPU_alloc_matrix(int rows, int cols){

    double *m;
    int i;

    if ((m = (double*)malloc(rows * cols * sizeof(double))) == NULL) {
        return(NULL);
    }

    for (i = 0; i < rows * cols; i++){
        m[i] = 0.0;
    }
    
    return(m);
}


void GPU_matrix_free_2D(double **M, int n_layers){

    int i;

    for (i=0; i < n_layers; ++i) {
        if (M[i] != NULL) {
            cudaFree(M[i]);
        }
    }
    cudaFree(M);
}

void GPU_matrix_free(double *m){

    if (m != NULL)
        free(m);
}

// void matrix_sum(double *c, double *a, double *b, int rows, int cols){

//     int  col, row;
//     double sum;

//     for (row = 0; row < rows; row++) {
//         for(col = 0; col < cols; col++) {
//             sum = *m_elem(a, cols, row, col) + *m_elem(b, cols, row, col);
//             //printf("- %f %f %f \n ", *m_elem(a, cols, row, col), *m_elem(b, cols, row, col),sum);
//             *m_elem(c, cols, row, col) = sum;
//         }
//     }
// }

// void matrix_sub(double *c, double *a, double *b, int rows, int cols){

//     int col, row;
//     double sum;

//     for (row = 0; row < rows; row++) {
//         for(col = 0; col < cols; col++) {
//             sum = *m_elem(a, cols, row, col) - *m_elem(b, cols, row, col);
//             *m_elem(c, cols, row, col) = sum;
//         }
//     }
// }

// void matrix_mul_cnt(double *m, int rows, int cols, double cnt){

//     int col, row;

//     for (row = 0; row < rows; row++) {
//         for(col = 0; col < cols; col++) {
//             *m_elem(m, cols, row, col) *= cnt;
//         }
//     }
// }

// void matrix_zero(double *m, int rows, int cols){

//     int col, row;

//     for (row = 0; row < rows; row++) {
//         for(col = 0; col < cols; col++) {
//             *m_elem(m, cols, row, col) = 0.0;
//         }
//     }
// }

// void matrix_mul_dot(double *c, double *a, double *b, int rows, int cols){

//     int col, row;
//     double prod;

//     for (row = 0; row < rows; row++) {
//         for(col = 0; col < cols; col++) {
//             prod = *m_elem(a, cols, row, col) * *m_elem(b, cols, row, col);
//             //printf("- %f %f %f \n ", *m_elem(a, rows, row, col), *m_elem(b, rows, row, col),sum);
//             *m_elem(c, cols, row, col) = prod;
//         }
//     }
// }

// double *matrix_transpose(double *m, int rows, int cols){

//     double *m_t;
//     int i, j;

//     if ((m_t = (double*)malloc(rows * cols * sizeof(double))) == NULL) {
//         return(NULL);
//     }

//     for (i = 0; i < rows; i++){
//         for (j = 0; j < cols; j++){
//             *m_elem(m_t, rows, j, i) = *m_elem(m, cols, i, j);
//         }
//     }
    
//     return(m_t);
// }

// void matrix_mul(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols){

//     assert(a_cols == b_rows);

//     int i, col, row;
//     double sum;

// #ifdef TIMING
//     int res_time;
//     struct timespec t1, t2;
//     clockid_t clk_id = CLOCK_MONOTONIC;
//     res_time = clock_gettime(clk_id, &t1);
// #endif

//     for (row = 0; row < a_rows; row++) {
//         for(col = 0; col < b_cols; col++) {
//             sum = 0.0;
//             for (i = 0; i < a_cols; i++) {
//                 sum += *m_elem(a, a_cols, row, i) * *m_elem(b, b_cols, i, col);
//                 //printf("%lf %lf\n", *m_elem(a, a_cols, row, i), *m_elem(b, b_cols, i, col));
//             }
//             *m_elem(c, b_cols, row, col) = sum;
//         }
//     }

// #ifdef TIMING
//     res_time = clock_gettime(clk_id, &t2);
//     printf("Matrix mul execution time: %ld us \n", diff_time(t2, t1));
// #endif

// }

void GPU_matrix_mul_add(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols, double *d){

    // int i, col, row;
    // double sum;
    double *A, *B, *C, *D;

    gpuErrchk(cudaMalloc(&A, a_rows * a_cols * sizeof(double)));
    gpuErrchk(cudaMalloc(&B, b_rows * b_cols * sizeof(double)));
    gpuErrchk(cudaMalloc(&C, a_rows * b_cols * sizeof(double)));
    gpuErrchk(cudaMalloc(&D, a_rows * b_cols * sizeof(double)));

    cudaMemcpy(A, a, a_rows * a_cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(B, b, b_rows * b_cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(D, d, a_rows * b_cols * sizeof(double), cudaMemcpyHostToDevice);

    
    dimBlock = dim3(512, 512);
    dimGrid = dim3((b_rows + dimBlock.x - 1) / dimBlock.x, (a_rows + dimBlock.y - 1) /dimBlock.y);

    mul_add<<<dimGrid, dimBlock>>>(c, a, b, a_rows, a_cols, b_rows, b_cols, d);

    cudaMemcpy(c, C, a_rows * b_cols * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(D);
}

void GPU_matrix_func(double *n, double *m, int rows, int cols, double (*func)(double)){

    double *N, *M;

    gpuErrchk(cudaMalloc(&N, rows * cols * sizeof(double)));
    gpuErrchk(cudaMalloc(&M, rows * cols * sizeof(double)));

    cudaMemcpy(N, n, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(M, m, rows * cols * sizeof(double), cudaMemcpyHostToDevice);



    // int col, row;

    // for (row = 0; row < rows; row++){
    //     for(col = 0; col < cols; col++){
    //         // *m_elem(n, cols, row, col) = func(*m_elem(m, cols, row, col));
    //     }
    // }
}

// void print_matrix(double *m, int m_rows, int m_cols){
    
//     int col, row;
//     printf("%d %d\n", m_rows, m_cols);
//     for (row = 0; row < m_rows; row++){
//         for(col = 0; col < m_cols; col++){
//             printf("(%d %d) %.*lf ", row, col, 10, *m_elem(m, m_cols, row, col));
//         }
//         printf("\n");
//     }
//     printf("\n");
// }