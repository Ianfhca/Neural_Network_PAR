#include <assert.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#include "globals.h"
#include "matrix.cuh"
#include "nn_aux.h"

#ifdef TIMING
#include <time.h>

#include "utils.h"
#endif

double **alloc_matrix_2v(int n_layers, int *size, int *size_prev, double (*init_weight_ptr)(void)) {
    double **m;
    int i, j;

    if ((m = (double **)malloc(n_layers * sizeof(double *))) == NULL) {
        return (NULL);
    }

    for (i = 0; i < n_layers; i++)
        if ((m[i] = (double *)malloc(size[i] * size_prev[i] * sizeof(double))) == NULL) {
            matrix_free_2D(m, n_layers);
            return (NULL);
        }

    for (i = 0; i < n_layers; i++) {
        for (j = 0; j < size[i] * size_prev[i]; j++) {
            m[i][j] = init_weight_ptr();
        }
    }

    return (m);
}

double **alloc_matrix_1v(int n_layers, int *size, double (*init_weight_ptr)(void)) {
    double **m;
    int i, j;

    if ((m = (double **)malloc(n_layers * sizeof(double *))) == NULL) {
        return (NULL);
    }

    for (i = 0; i < n_layers; i++)
        if ((m[i] = (double *)malloc(size[i] * sizeof(double))) == NULL) {
            matrix_free_2D(m, n_layers);
            return (NULL);
        }

    for (i = 0; i < n_layers; i++) {
        for (j = 0; j < size[i]; j++) {
            m[i][j] = init_weight_ptr();
        }
    }

    return (m);
}

double *alloc_array(int length) {
    double *v;
    int i;

    if ((v = (double *)malloc(length * sizeof(double))) == NULL) {
        return (NULL);
    }

    for (i = 0; i < length; i++) {
        v[i] = 0.0;
    }

    return (v);
}

double *alloc_matrix(int rows, int cols) {
    double *m;
    int i;

    if ((m = (double *)malloc(rows * cols * sizeof(double))) == NULL) {
        return (NULL);
    }

    for (i = 0; i < rows * cols; i++) {
        m[i] = 0.0;
    }

    return (m);
}

void matrix_free_2D(double **m, int n_layers) {
    int i;

    for (i = 0; i < n_layers; ++i) {
        if (m[i] != NULL) {
            free(m[i]);
        }
    }
    free(m);
}

void matrix_free(double *m) {
    if (m != NULL)
        free(m);
}

double *m_elem(double *m, int length, int x, int y) {
    return (double *)&m[length * x + y];
}

void matrix_sum(double *c, double *a, double *b, int rows, int cols) {
    int col, row;
    double sum;

    for (row = 0; row < rows; row++) {
        for (col = 0; col < cols; col++) {
            sum = *m_elem(a, cols, row, col) + *m_elem(b, cols, row, col);
            // printf("- %f %f %f \n ", *m_elem(a, cols, row, col), *m_elem(b, cols, row, col),sum);
            *m_elem(c, cols, row, col) = sum;
        }
    }
}

void matrix_sub(double *c, double *a, double *b, int rows, int cols) {
    int col, row;
    double sum;

    for (row = 0; row < rows; row++) {
        for (col = 0; col < cols; col++) {
            sum = *m_elem(a, cols, row, col) - *m_elem(b, cols, row, col);
            *m_elem(c, cols, row, col) = sum;
        }
    }
}

void matrix_mul_cnt(double *m, int rows, int cols, double cnt) {
    int col, row;

    for (row = 0; row < rows; row++) {
        for (col = 0; col < cols; col++) {
            *m_elem(m, cols, row, col) *= cnt;
        }
    }
}

void matrix_zero(double *m, int rows, int cols) {
    int col, row;

    for (row = 0; row < rows; row++) {
        for (col = 0; col < cols; col++) {
            *m_elem(m, cols, row, col) = 0.0;
        }
    }
}

void matrix_mul_dot(double *c, double *a, double *b, int rows, int cols) {
    int col, row;
    double prod;

    for (row = 0; row < rows; row++) {
        for (col = 0; col < cols; col++) {
            prod = *m_elem(a, cols, row, col) * *m_elem(b, cols, row, col);
            // printf("- %f %f %f \n ", *m_elem(a, rows, row, col), *m_elem(b, rows, row, col),sum);
            *m_elem(c, cols, row, col) = prod;
        }
    }
}

double *matrix_transpose(double *m, int rows, int cols) {
    double *m_t;
    int i, j;

    if ((m_t = (double *)malloc(rows * cols * sizeof(double))) == NULL) {
        return (NULL);
    }

    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            *m_elem(m_t, rows, j, i) = *m_elem(m, cols, i, j);
        }
    }

    return (m_t);
}

void matrix_mul(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols) {
    assert(a_cols == b_rows);

    int i, col, row;
    double sum;

#ifdef TIMING
    int res_time;
    struct timespec t1, t2;
    clockid_t clk_id = CLOCK_MONOTONIC;
    res_time = clock_gettime(clk_id, &t1);
#endif

    for (row = 0; row < a_rows; row++) {
        for (col = 0; col < b_cols; col++) {
            sum = 0.0;
            for (i = 0; i < a_cols; i++) {
                sum += *m_elem(a, a_cols, row, i) * *m_elem(b, b_cols, i, col);
                // printf("%lf %lf\n", *m_elem(a, a_cols, row, i), *m_elem(b, b_cols, i, col));
            }
            *m_elem(c, b_cols, row, col) = sum;
        }
    }

#ifdef TIMING
    res_time = clock_gettime(clk_id, &t2);
    printf("Matrix mul execution time: %ld us \n", diff_time(t2, t1));
#endif
}

void matrix_mul_add(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols, double *d) {
    int i, col, row;
    double sum;

    for (row = 0; row < a_rows; row++) {
        for (col = 0; col < b_cols; col++) {
            sum = 0.0;
            for (i = 0; i < a_cols; i++) {
                sum += *m_elem(a, a_cols, row, i) * *m_elem(b, b_cols, i, col);
                // printf("%lf %lf\n", *m_elem(a, a_cols, row, i), *m_elem(b, b_cols, i, col));
            }
            *m_elem(c, b_cols, row, col) = sum + *m_elem(d, b_cols, row, col);
        }
    }
}

void matrix_func(double *n, double *m, int rows, int cols, double (*func)(double)) {
    int col, row;

    for (row = 0; row < rows; row++) {
        for (col = 0; col < cols; col++) {
            *m_elem(n, cols, row, col) = func(*m_elem(m, cols, row, col));
        }
    }
}

void print_matrix(double *m, int m_rows, int m_cols) {
    int col, row;
    printf("%d %d\n", m_rows, m_cols);
    for (row = 0; row < m_rows; row++) {
        for (col = 0; col < m_cols; col++) {
            printf("(%d %d) %.*lf ", row, col, 10, *m_elem(m, m_cols, row, col));
        }
        printf("\n");
    }
    printf("\n");
}

// ----------------------------------- Funciones CUDA -----------------------------------

// void cuda_matrix_mul_add(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols, double *d) {
//     int i, col, row;
//     double sum;

//     for (row = 0; row < a_rows; row++) {
//         for (col = 0; col < b_cols; col++) {
//             sum = 0.0;
//             for (i = 0; i < a_cols; i++) {
//                 sum += *m_elem(a, a_cols, row, i) * *m_elem(b, b_cols, i, col);
//                 // printf("%lf %lf\n", *m_elem(a, a_cols, row, i), *m_elem(b, b_cols, i, col));
//             }
//             *m_elem(c, b_cols, row, col) = sum + *m_elem(d, b_cols, row, col);
//         }
//     }
// }

// https://stackoverflow.com/questions/5885195/using-cudamalloc-to-allocate-a-matrix/5885346#5885346

double **cuda_alloc_matrix_1v(int n_layers, int *size, double (*init_weight_ptr)(void), int size_batch) {
    double **m;
    int i, j;
    cudaError_t err = cudaGetLastError();

    // Asignar memoria para el arreglo de punteros a filas en el host
    if ((err = cudaHostAlloc((void **)&m, n_layers * sizeof(double *), cudaHostAllocDefault)) != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return NULL;
    }

    for (int i = 0; i < n_layers; i++) {
        // Asignar memoria para cada fila en el host
        if ((err = cudaHostAlloc((void **)&(m[i]), size[i] * size_batch * sizeof(double), cudaHostAllocDefault)) != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
            return NULL;
        }
    }

    for (i = 0; i < n_layers; i++) {
        for (j = 0; j < size[i] * size_batch; j++) {
            m[i][j] = init_weight_ptr();
        }
    }

    return m;
}

double **cuda_alloc_matrix_2v(int n_layers, int *size, int *size_prev, double (*init_weight_ptr)(void), int size_batch) { 
    double **m;
    int i, j;
    cudaError_t err = cudaGetLastError();

    // Asignar memoria para el arreglo de punteros a filas en el host
    if ((err = cudaHostAlloc((void **)&m, n_layers * sizeof(double *), cudaHostAllocDefault)) != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return NULL;
    }

    for (int i = 0; i < n_layers; i++) {
        // Asignar memoria para cada fila en el host
        if ((err = cudaHostAlloc((void **)&(m[i]), size[i] * size_prev[i] * size_batch * sizeof(double), cudaHostAllocDefault)) != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
            return NULL;
        }
    }

    for (i = 0; i < n_layers; i++) {
        for (j = 0; j < size[i] * size_batch * size_prev[i]; j++) {
            m[i][j] = init_weight_ptr();
        }
    }

    return m;
}

// https://stackoverflow.com/questions/38978297/pointer-to-array-of-pointers-to-objects-in-cuda

// -- Version 2 -- Devuelve la matriz de la GPU (double **M)

// double **cuda_alloc_matrix_1v(int n_layers, int *size, double (*init_weight_ptr)(void)) {
//     double **m, **M;
//     int i, j;
//     cudaError_t err = cudaGetLastError();

//     if ((m = (double **)malloc(n_layers * sizeof(double *))) == NULL) {
//         return (NULL);
//     }

//     if ((err = cudaMalloc((void ***)&M, (n_layers * sizeof(double)))) != cudaSuccess) {
//         return (NULL);
//     }

//     for (i = 0; i < n_layers; i++) {
//         if ((m[i] = (double *)malloc(size[i] * sizeof(double))) == NULL) {
//             matrix_free_2D(m, n_layers);
//             return (NULL);
//         }
//     }

//     for (i = 0; i < n_layers; i++) {
//         for (j = 0; j < size[i]; j++) {
//             m[i][j] = init_weight_ptr();
//         }
//         if ((err = cudaMalloc((void **)&(m[i]), size[i] * sizeof(double))) != cudaSuccess) {
//             cuda_matrix_free_2D(m, n_layers);
//             printf("CUDA error: %s\n", cudaGetErrorString(err));
//             return (NULL);
//         }
//         cudaMemcpy(m[i], m[i], size[i] * sizeof(double), cudaMemcpyHostToDevice);
//     }
//     cudaMemcpy(M, m, n_layers * sizeof(double), cudaMemcpyHostToDevice);

//     return (M);
// }

// --Version 1 -- Devuelve una matriz con un solo puntero

// double *cuda_alloc_matrix_1v(int n_layers, int *size, double (*init_weight_ptr)(void)) {
//     double *m, *M;
//     int total_size = 0;
//     int i, j;
//     cudaError_t err = cudaGetLastError();

//     for (i = 0; i < n_layers; i++) {
//         for (j = 0; j < size[i]; j++) {
//             total_size++;
//         }
//     }

//     if ((m = (double *)malloc(total_size * sizeof(double))) == NULL) {
//         return (NULL);
//     }

//     if ((err = cudaMalloc(&M, total_size * sizeof(double))) != cudaSuccess) {
//         printf("CUDA error: %s\n", cudaGetErrorString(err));
//         return (NULL);
//     }

//     for (i = 0; i < n_layers; i++) {
//         for (j = 0; j < size[i]; j++) {
//             m[i * size[i] + j] = 0;  // init_weight_ptr();
//             printf("%lf ", m[i * size[i] + j]);
//         }
//         printf("\n");
//     }

//     if ((err = cudaMemcpy(M, m, total_size * sizeof(double), cudaMemcpyHostToDevice)) != cudaSuccess) {
//         printf("CUDA error: %s\n", cudaGetErrorString(err));
//         return (NULL);
//     }
//     printf("-- PUNTO DE CONTROL --\n");
//     return (M);
// }
