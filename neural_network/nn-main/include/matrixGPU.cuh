#ifndef __GPU_MATRIX_H
#define __GPU_MATRIX_H

#include <stdbool.h>

#define THR_PER_BLOCK 1024

#define gpuErrchk(call)                                 \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error: %s\n", cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)

double **GPU_alloc_matrix_1v(int n_layers, int *size, double (*init_weight_ptr)(void));

double **GPU_alloc_matrix_2v(int n_layers, int *size, int *size_prev, double (*init_weight_ptr)(void));

double *GPU_alloc_array(int length);

double *GPU_alloc_matrix(int rows, int cols);

void GPU_matrix_free_2D(double **m, int n_layers);

void GPU_matrix_free(double *m);

double *GPU_m_elem(double *m, int length, int x, int y);

// void GPU_matrix_sum(double *c, double *a, double *b, int rows, int cols);

// void GPU_matrix_sub(double *c, double *a, double *b, int rows, int cols);

// void GPU_matrix_mul_cnt(double *m, int rows, int cols, double cnt);

// void GPU_matrix_zero(double *m, int rows, int cols);

// void GPU_matrix_mul_dot(double *c, double *a, double *b, int rows, int cols);

// double *GPU_matrix_transpose(double *m, int rows, int cols);

// void GPU_matrix_mul(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols);

// void GPU_matrix_mul_trans(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols);

void GPU_matrix_mul_add(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols, double* d);

void GPU_matrix_func(double *n, double *m, int m_rows, int m_cols, double (*func)(double));

// void GPU_print_matrix(double *m, int m_rows, int m_cols);

#endif
