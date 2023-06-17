#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "ds.h"
#include "globals.h"
#include "matrix.cuh"
#include "nn.cuh"
#include "nn_aux.h"
#include "test.h"
#include "train.h"
#include "train_GPU.cuh"
#include "utils.h"
// #include "train_GPU.cu"

void init_nn(nn_t *nn, int n_layers, int *layers_size) {
    int i;

    nn->n_layers = n_layers;
    nn->layers_size = layers_size;
    nn->init_weight_ptr = init_weight_rnd;
    nn->activation_ptr = (activation_ptr_t *)malloc((nn->n_layers - 1) * sizeof(activation_ptr_t));
    nn->dactivation_ptr = (activation_ptr_t *)malloc((nn->n_layers - 1) * sizeof(activation_ptr_t));
    for (i = 0; i < n_layers - 1; i++) {
        nn->activation_ptr[i] = sigmoid;
        nn->dactivation_ptr[i] = dSigmoid;
    }
    nn->loss = mse;
#ifdef CPU
    nn->BH = alloc_matrix_1v(n_layers - 1, &layers_size[1], nn->init_weight_ptr);
    nn->WH = alloc_matrix_2v(n_layers - 1, &layers_size[1], &layers_size[0], nn->init_weight_ptr);
#else
    nn->BH = cuda_alloc_matrix_1v(n_layers - 1, &layers_size[1], nn->init_weight_ptr, 1);
    nn->WH = cuda_alloc_matrix_2v(n_layers - 1, &layers_size[1], &layers_size[0], nn->init_weight_ptr, 1);
#endif
}

// #ifdef CPU

void train(nn_t *nn, ds_t *ds, int epochs, int size_batch, double lr) {
    int i, n, x, n_batches, min_batch;
    double **A, **Z, **D, **d;
    int *order;
    double loss;
    struct timespec t1, t2;
    clockid_t clk_id = CLOCK_MONOTONIC;

    order = (int *)malloc(ds->n_samples * sizeof(int));

    A = alloc_matrix_1v(nn->n_layers, nn->layers_size, init_zero);
    Z = alloc_matrix_1v(nn->n_layers, nn->layers_size, init_zero);
    D = alloc_matrix_2v(nn->n_layers - 1, &(nn->layers_size[1]), &(nn->layers_size[0]), init_zero);
    d = alloc_matrix_1v(nn->n_layers - 1, &(nn->layers_size[1]), init_zero);

    n_batches = ds->n_samples / size_batch;
    // printf("Num_batches: %d\n", n_batches);

    for (i = 0; i < ds->n_samples; i++)
        order[i] = i;

    for (n = 0; n < epochs; n++) {
        if (verbose)
            printf("Epoch %d/%d \n", n, epochs);

        loss = 0.0;
        shuffle(order, ds->n_samples);

        clock_gettime(clk_id, &t1);

        for (x = 0; x < n_batches; x++) {
            for (min_batch = (x * size_batch); min_batch < ((x + 1) * size_batch); min_batch++) {
                i = order[min_batch];
                forward_pass(nn, &ds->inputs[i * ds->n_inputs], A, Z);
                loss += back_prop(nn, &ds->outputs[i * ds->n_outputs], A, Z, D, d);
            }

            update(nn, D, d, lr, size_batch);
        }

        clock_gettime(clk_id, &t2);

        if (verbose)
            printf(" time: %ld us - loss: %.*f\n", diff_time(t2, t1), 12, loss / ds->n_samples);
    }
}

void test(nn_t *nn, ds_t *ds) {
    int i;
    double **A;
    int tp, tn, fp, fn;
    float p, r, f;

    A = alloc_matrix_1v(nn->n_layers, nn->layers_size, init_zero);

    for (i = 0; i < ds->n_samples; i++) {
        forward_pass_test(nn, &ds->inputs[i * ds->n_inputs], A);
        if (ds->outputs[0] == 0 && round(A[nn->n_layers - 1][0]) == 1) {
            fn++;
        } else if (ds->outputs[0] == 0 && round(A[nn->n_layers - 1][0]) == 0) {
            tn++;
        } else if (ds->outputs[0] == 1 && round(A[nn->n_layers - 1][0]) == 0) {
            fp++;
        } else if (ds->outputs[0] == 1 && round(A[nn->n_layers - 1][0]) == 1) {
            tp++;
        }
    }

    // Precision
    p = precision(tp, fp);
    // Recall
    r = recall(tp, fn);
    // F1
    f = f1(p, r);

    printf("Precision: %f, Recall: %f, f1: %f\n", p, r, f);
}
// #else
// #elif defined(GPU)
// #include "train_GPU.cu"

__device__ void cuda_matrix_free_2D(double **m, int n_layers) {
    int i;

    for (i = 0; i < n_layers; ++i) {
        if (m[i] != NULL) {
            free(m[i]);
        }
    }
    free(m);
}

__device__ void cuda_matrix_free(double *m) {
    if (m != NULL)
        free(m);
}

__device__ void cuda_matrix_mul(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols) {
    if (a_cols != b_rows) return;

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
                sum += a[a_cols * row + i] * b[b_cols * i + col];
            }
            c[b_cols * row + col] = sum;
        }
    }

#ifdef TIMING
    res_time = clock_gettime(clk_id, &t2);
    printf("Matrix mul execution time: %ld us \n", diff_time(t2, t1));
#endif
}

__device__ void cuda_matrix_transpose_mul1(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols) {
    if (a_cols != b_rows) return;

    int i, col, row;
    double sum;

#ifdef TIMING
    int res_time;
    struct timespec t1, t2;
    clockid_t clk_id = CLOCK_MONOTONIC;
    res_time = clock_gettime(clk_id, &t1);
#endif

    // for (row = 0; row < a_rows; row++) {
    //     for (col = 0; col < b_cols; col++) {
    //         sum = 0.0;
    //         for (i = 0; i < a_cols; i++) {
    //             sum += a[a_cols * row + i] * b[b_cols * i + col];
    //         }
    //         c[b_cols * row + col] = sum;
    //     }
    // }
    for (row = 0; row < a_rows; row++) {
        for (col = 0; col < b_rows; col++) {
            sum = 0.0;
            for (i = 0; i < a_cols; i++) {
                sum += a[a_cols * row + i] * b[b_cols * col + i];
            }
            c[b_rows * row + col] = sum;
        }
    }

#ifdef TIMING
    res_time = clock_gettime(clk_id, &t2);
    printf("Matrix mul execution time: %ld us \n", diff_time(t2, t1));
#endif
}

__device__ void cuda_matrix_transpose_mul2(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols) {
    if (a_cols != b_rows) return;

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
                sum += a[a_cols * i + row] * b[b_cols * i + col];
            }
            c[b_cols * row + col] = sum;
        }
    }

#ifdef TIMING
    res_time = clock_gettime(clk_id, &t2);
    printf("Matrix mul execution time: %ld us \n", diff_time(t2, t1));
#endif
}

__device__ void cuda_matrix_mul_dot(double *c, double *a, double *b, int rows, int cols) {
    int col, row;
    double prod;

    for (row = 0; row < rows; row++) {
        for (col = 0; col < cols; col++) {
            prod = a[cols * row + col] * b[cols * row + col];
            c[cols * row + col] = prod;
        }
    }
}

__device__ void cuda_matrix_sum(double *c, double *a, double *b, int rows, int cols) {
    int col, row;
    double sum;

    for (row = 0; row < rows; row++) {
        for (col = 0; col < cols; col++) {
            sum = a[cols * row + col] + b[cols * row + col];
            c[cols * row + col] = sum;
        }
    }
}

__device__ void cuda_matrix_sub(double *c, double *a, double *b, int rows, int cols) {
    int col, row;
    double sum;

    for (row = 0; row < rows; row++) {
        for (col = 0; col < cols; col++) {
            sum = a[cols * row + col] - b[cols * row + col];
            c[cols * row + col] = sum;
        }
    }
}

// -- Funciones de los punteros --
__device__ double cuda_mse(double *a, double *output, int length) {
    int i;
    double cost = 0.0;

    for (i = 0; i < length; i++) {
        cost += ((a[i] - output[i]) * (a[i] - output[i]));
    }
    cost /= length;

    return (cost);
}

__device__ double cuda_dSigmoid(double x) {
    double sig_z = 1.0 / (1.0 + exp(-x));
    return (sig_z * (1 - sig_z));
}

__device__ double cuda_sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

__device__ double *cuda_m_elem(double *m, int length, int x, int y) {
    return (double *)&m[length * x + y];
}

// -- Punteros a las funciones --
__device__ loss_t cuda_mse_ptr = cuda_mse;
__device__ activation_ptr_t cuda_dSigmoid_ptr = cuda_dSigmoid;
__device__ activation_ptr_t cuda_sigmoid_ptr = cuda_sigmoid;

// --------------------------- OMEGA-KERNEL ---------------------------
__global__ void batch_process(nn_t *nn, ds_t *ds, int x, int size_batch, int *order, double *loss, double **A, double **Z, double **D, double **d, double **BH, double **WH, double **D_aux, double **E) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i, j, o;

    if (tid >= (x * size_batch) && tid < ((x + 1) * size_batch)) {
        o = order[(x * size_batch) + tid];
        double *input = &ds->inputs[o * ds->n_inputs];

        // FUNCION forward_pass_GPU:
        for (i = 0; i < nn->layers_size[0]; i++) {
            A[0][(tid * size_batch) + i] = input[i];
        }

        for (i = 1; i < nn->n_layers; i++) {
            // FUNCION matrix_mul_add
            int col, row;
            int a_rows, acols_brows, b_cols;
            double sum;

            a_rows = nn->layers_size[i];
            acols_brows = nn->layers_size[i - 1];
            b_cols = 1;

            for (row = 0; row < a_rows; row++) {
                for (col = 0; col < b_cols; col++) {
                    sum = 0.0;
                    for (j = 0; j < acols_brows; j++) {
                        sum += WH[i - 1][acols_brows * row + j] * A[i - 1][(size_batch * tid) + (b_cols * j + col)];
                    }
                    Z[i][(size_batch * tid) + (b_cols * row + col)] = BH[i - 1][b_cols * row + col] + sum;
                }
            }

            // FUNCION matrix_func -> activation_ptr
            int rows, cols;
            rows = nn->layers_size[i];
            cols = 1;

            for (row = 0; row < rows; row++) {
                for (col = 0; col < cols; col++) {
                    A[i][(size_batch * tid) + (cols * row + col)] = nn->activation_ptr[i - 1](Z[i][(size_batch * tid) + (cols * row + col)]);
                }
            }

            // FUNCION matrix_func -> dactivation_ptr
            for (row = 0; row < rows; row++) {
                for (col = 0; col < cols; col++) {
                    Z[i][(size_batch * tid) + (cols * row + col)] = nn->dactivation_ptr[i - 1](A[i][(size_batch * tid) + (cols * row + col)]);
                }
            }
        }

        // FUNCION back_prop
        int n_l;
        int *l_s;
        double local_loss = 0;

        n_l = nn->n_layers;
        l_s = nn->layers_size;

        local_loss = nn->loss(&A[n_l - 1][size_batch * tid], &ds->outputs[o * ds->n_outputs], l_s[n_l - 1]);
        // printf("%lf\n", local_loss);

        cuda_matrix_sub(&E[n_l - 2][size_batch * tid], &A[n_l - 1][size_batch * tid], &ds->outputs[o * ds->n_outputs], l_s[n_l - 1], 1);
        cuda_matrix_mul_dot(&E[n_l - 2][size_batch * tid], &E[n_l - 2][size_batch * tid], &Z[n_l - 1][size_batch * tid], l_s[n_l - 1], 1);

        cuda_matrix_transpose_mul1(&D_aux[n_l - 2][size_batch * tid], &E[n_l - 2][size_batch * tid], &A[n_l - 2][size_batch * tid], l_s[n_l - 1], 1, 1, l_s[n_l - 2]);

        cuda_matrix_sum(&D[n_l - 2][size_batch * tid], &D[n_l - 2][size_batch * tid], &D_aux[n_l - 2][size_batch * tid], l_s[n_l - 1], l_s[n_l - 2]);
        cuda_matrix_sum(&d[n_l - 2][size_batch * tid], &d[n_l - 2][size_batch * tid], &E[n_l - 2][size_batch * tid], l_s[n_l - 1], 1);

        for (i = n_l - 2; i > 0; i--) {
            cuda_matrix_transpose_mul2(&E[i - 1][size_batch * tid], WH[i], E[i], l_s[i], l_s[i + 1], l_s[i + 1], 1);

            cuda_matrix_mul_dot(&E[i - 1][size_batch * tid], &E[i - 1][size_batch * tid], &Z[i][size_batch * tid], l_s[i], 1);

            cuda_matrix_mul(&D_aux[i - 1][size_batch * tid], &E[i - 1][size_batch * tid], &A[i - 1][size_batch * tid], l_s[i], 1, 1, l_s[i - 1]);

            cuda_matrix_sum(&D[i - 1][size_batch * tid], &D[i - 1][size_batch * tid], &D_aux[i - 1][size_batch * tid], l_s[i], l_s[i - 1]);
            cuda_matrix_sum(&d[i - 1][size_batch * tid], &d[i - 1][size_batch * tid], &E[i - 1][size_batch * tid], l_s[i], 1);
        }

        // cuda_matrix_free_2D(D_aux, n_l - 1);
        // cuda_matrix_free_2D(E, n_l - 1);

        // printf("loss[%d]: %lf", tid, local_loss);
        // atomicAdd(loss, local_loss);

        printf("Llegamos\n");
    }
}

void train_GPU(nn_t *nn, ds_t *ds, int epochs, int size_batch, double lr) {
    int i, n, x, n_batches;
    double **A, **Z, **D, **d;
    double **D_aux, **E;
    nn_t *d_nn;
    ds_t *d_ds;
    int *order, *d_order;
    double loss;
    struct timespec t1, t2;
    clockid_t clk_id = CLOCK_MONOTONIC;
    cudaError_t error;

    // Reserva de las matrices ----------------------------------------------------------------
    A = cuda_alloc_matrix_1v(nn->n_layers, nn->layers_size, init_zero, size_batch);
    Z = cuda_alloc_matrix_1v(nn->n_layers, nn->layers_size, init_zero, size_batch);
    D = cuda_alloc_matrix_2v(nn->n_layers - 1, &(nn->layers_size[1]), &(nn->layers_size[0]), init_zero, size_batch);
    d = cuda_alloc_matrix_1v(nn->n_layers - 1, &(nn->layers_size[1]), init_zero, size_batch);

    D_aux = cuda_alloc_matrix_2v(nn->n_layers - 1, &(nn->layers_size[1]), &(nn->layers_size[0]), init_zero, size_batch);
    E = cuda_alloc_matrix_1v(nn->n_layers - 1, &(nn->layers_size[1]), init_zero, size_batch);

    // Reserva de la estructura nn ------------------------------------------------------------
    cudaMalloc((void **)&d_nn, sizeof(nn_t));

    cudaMemcpy(&(d_nn->n_layers), &(nn->n_layers), sizeof(int), cudaMemcpyHostToDevice);

    int *d_layers_size;
    cudaMalloc((void **)&d_layers_size, nn->n_layers * sizeof(int));
    cudaMemcpy(d_layers_size, nn->layers_size, nn->n_layers * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_nn->layers_size), &d_layers_size, sizeof(int *), cudaMemcpyHostToDevice);

    activation_ptr_t *h_func_vec = (activation_ptr_t *)malloc((nn->n_layers - 1) * sizeof(activation_ptr_t));
    for (i = 0; i < (nn->n_layers - 1); i++) {
        error = cudaMemcpyFromSymbol(&(h_func_vec[i]), cuda_sigmoid_ptr, sizeof(activation_ptr_t));
        if (error != cudaSuccess) {
            printf("Cuda error copying activation_ptr: %s\n", cudaGetErrorString(error));
        }
    }

    activation_ptr_t *d_activation_ptr;
    cudaMalloc((void **)&d_activation_ptr, (nn->n_layers - 1) * sizeof(activation_ptr_t));
    cudaMemcpy(d_activation_ptr, h_func_vec, (nn->n_layers - 1) * sizeof(activation_ptr_t), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_nn->activation_ptr), &d_activation_ptr, sizeof(activation_ptr_t *), cudaMemcpyHostToDevice);

    for (i = 0; i < (nn->n_layers - 1); i++) {
        error = cudaMemcpyFromSymbol(&(h_func_vec[i]), cuda_sigmoid_ptr, sizeof(activation_ptr_t));
        if (error != cudaSuccess) {
            printf("Cuda error copying dactivation_ptr: %s\n", cudaGetErrorString(error));
        }
    }

    activation_ptr_t *d_dactivation_ptr;
    cudaMalloc((void **)&d_dactivation_ptr, (nn->n_layers - 1) * sizeof(activation_ptr_t));
    cudaMemcpy(d_dactivation_ptr, h_func_vec, (nn->n_layers - 1) * sizeof(activation_ptr_t), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_nn->dactivation_ptr), &d_dactivation_ptr, sizeof(activation_ptr_t *), cudaMemcpyHostToDevice);

    free(h_func_vec);
    // cudaFree(d_activation_ptr);
    // cudaFree(d_dactivation_ptr);

    error = cudaMemcpyFromSymbol(&d_nn->loss, cuda_mse_ptr, sizeof(loss_t));
    if (error != cudaSuccess) {
        printf("Cuda error copying loss: %s\n", cudaGetErrorString(error));
    }

    // Reserva de la estructura ds ------------------------------------------------------------
    cudaMalloc((void **)&d_ds, sizeof(ds_t));

    cudaMemcpy(&(d_ds->n_samples), &(ds->n_samples), sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_ds->n_inputs), &(ds->n_inputs), sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_ds->n_outputs), &(ds->n_outputs), sizeof(int), cudaMemcpyHostToDevice);

    double *d_inputs;
    cudaMalloc((void **)&d_inputs, ds->n_inputs * ds->n_samples * sizeof(double));
    cudaMemcpy(d_inputs, ds->inputs, ds->n_inputs * ds->n_samples * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_ds->inputs), &d_inputs, sizeof(double *), cudaMemcpyHostToDevice);

    double *d_outputs;
    cudaMalloc((void **)&d_outputs, ds->n_outputs * ds->n_samples * sizeof(double));
    cudaMemcpy(d_outputs, ds->outputs, ds->n_outputs * ds->n_samples * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_ds->outputs), &d_outputs, sizeof(double *), cudaMemcpyHostToDevice);

    double *d_max;
    cudaMalloc((void **)&d_max, ds->n_inputs * sizeof(double));
    cudaMemcpy(d_max, ds->max, ds->n_inputs * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_ds->max), &d_max, sizeof(double *), cudaMemcpyHostToDevice);

    double *d_min;
    cudaMalloc((void **)&d_min, ds->n_inputs * sizeof(double));
    cudaMemcpy(d_min, ds->min, ds->n_inputs * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_ds->min), &d_min, sizeof(double *), cudaMemcpyHostToDevice);

    double *d_std;
    cudaMalloc((void **)&d_std, ds->n_inputs * sizeof(double));
    cudaMemcpy(d_std, ds->std, ds->n_inputs * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_ds->std), &d_std, sizeof(double *), cudaMemcpyHostToDevice);

    double *d_mean;
    cudaMalloc((void **)&d_mean, ds->n_inputs * sizeof(double));
    cudaMemcpy(d_mean, ds->mean, ds->n_inputs * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_ds->mean), &d_mean, sizeof(double *), cudaMemcpyHostToDevice);

    // Reserva del vector order -----------------------------------------------------------
    order = (int *)malloc(ds->n_samples * sizeof(int));
    cudaMalloc(&d_order, ds->n_samples * sizeof(int));

    n_batches = ds->n_samples / size_batch;

    for (i = 0; i < ds->n_samples; i++)
        order[i] = i;

    for (n = 0; n < epochs; n++) {
        if (verbose)
            printf("Epoch %d/%d \n", n, epochs);

        loss = 0.0;
        shuffle(order, ds->n_samples);

        int thr_per_blk = THR_PER_BLOCK;
        int blk_in_grid = ceil((float)n_batches / thr_per_blk);

        cudaMemcpy(d_order, order, ds->n_samples * sizeof(int), cudaMemcpyHostToDevice);

        clock_gettime(clk_id, &t1);
        for (x = 0; x < n_batches; x++) {
            batch_process<<<blk_in_grid, thr_per_blk>>>(d_nn, d_ds, x, size_batch, d_order, &loss, A, Z, D, d, nn->BH, nn->WH, D_aux, E);
            cudaDeviceSynchronize();
            // update(nn, D, d, lr, size_batch);
        }
        clock_gettime(clk_id, &t2); 

        if (verbose)
            printf(" time: %ld us - loss: %.*f\n", diff_time(t2, t1), 12, loss / ds->n_samples);
    }
    // cudaFree(A);
    // cudaFree(Z);
    // cudaFree(D);
    // cudaFree(d);
    cudaFree(d_order);
    printf("-- PUNTO DE CONTROL --\n");
}

void test_GPU(nn_t *nn, ds_t *ds) {
    int i;
    double **A;

    A = alloc_matrix_1v(nn->n_layers, nn->layers_size, init_zero);

    for (i = 0; i < ds->n_samples; i++) {
        forward_pass_test(nn, &ds->inputs[i * ds->n_inputs], A);
    }

    // Precision
    // Recall
    // F1
}
// #endif

void print_nn(nn_t *nn) {
    int i, j, k;

    printf("Layers (I/H/O)\n");

    for (i = 0; i < nn->n_layers; i++) {
        printf("%d ", nn->layers_size[i]);
    }
    printf("\n");

    printf("Hidden Biases\n ");

    for (i = 0; i < nn->n_layers - 1; i++) {
        for (j = 0; j < nn->layers_size[i + 1]; j++) {
            printf("%lf ", nn->BH[i][j]);
        }
        printf("\n");
    }

    printf("Hidden Weights\n ");

    for (i = 0; i < nn->n_layers - 1; i++) {
        for (j = 0; j < nn->layers_size[i + 1]; j++) {
            for (k = 0; k < nn->layers_size[i]; k++) {
                printf("%lf ", nn->WH[i][(j * nn->layers_size[i]) + k]);
            }
            printf("\n");
        }
    }
}

void import_nn(nn_t *nn, char *filename) {
    int i, j, k;
    FILE *fd;

    if ((fd = fopen(filename, "r")) == NULL) {
        perror("Error importing the model\n");
        exit(1);
    }

    fscanf(fd, "%d ", &n_layers);

    layers = (int *)malloc(n_layers * sizeof(int));

    for (i = 0; i < n_layers; i++) {
        fscanf(fd, "%d ", &(layers[i]));
    }

    init_nn(nn, n_layers, layers);

    for (i = 0; i < nn->n_layers - 1; i++) {
        for (j = 0; j < nn->layers_size[i + 1]; j++) {
            fscanf(fd, "%lf ", &(nn->BH[i][j]));
        }
    }

    for (i = 0; i < nn->n_layers - 1; i++) {
        for (j = 0; j < nn->layers_size[i + 1]; j++) {
            for (k = 0; k < nn->layers_size[i]; k++) {
                fscanf(fd, "%lf ", &(nn->WH[i][(j * nn->layers_size[i]) + k]));
            }
        }
    }
    fclose(fd);
}

void export_nn(nn_t *nn, char *filename) {
    int i, j, k;
    FILE *fd;

    if ((fd = fopen(filename, "w")) == NULL) {
        perror("Error exporting the model");
        exit(1);
    }

    fprintf(fd, "%d\n", nn->n_layers);

    for (i = 0; i < nn->n_layers; i++) {
        fprintf(fd, "%d ", nn->layers_size[i]);
    }
    fprintf(fd, "\n");

    for (i = 0; i < nn->n_layers - 1; i++) {
        for (j = 0; j < nn->layers_size[i + 1]; j++) {
            fprintf(fd, "%lf ", nn->BH[i][j]);
        }
        fprintf(fd, "\n");
    }

    for (i = 0; i < nn->n_layers - 1; i++) {
        for (j = 0; j < nn->layers_size[i + 1]; j++) {
            for (k = 0; k < nn->layers_size[i]; k++) {
                fprintf(fd, "%lf ", nn->WH[i][(j * nn->layers_size[i]) + k]);
            }
            fprintf(fd, "\n");
        }
    }
    fclose(fd);
}
