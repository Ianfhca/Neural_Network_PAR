#ifndef __NN_H
#define __NN_H

#define THR_PER_BLOCK 1024
#define BLK_SIZE 32

#include <cuda.h>
#include "ds.h"

typedef double (*activation_ptr_t)(double);
typedef double (*loss_t)(double *a, double *output, int length);
typedef double (*init_weight_ptr_t)(void);

typedef struct nn_t {
    int n_layers;
    int *layers_size;

    double **WH;
    double **BH;

    loss_t loss;
    init_weight_ptr_t init_weight_ptr;
    activation_ptr_t *activation_ptr;
    activation_ptr_t *dactivation_ptr;

} nn_t;

// typedef struct p_nn_t {
//     int n_layers;
//     int *layers_size;

//     double *WH;
//     double *BH;

//     double (*loss)(double *a, double *output, int length);
//     double (*init_weight_ptr)(void);
//     activation_ptr_t *activation_ptr;
//     activation_ptr_t *dactivation_ptr;

// } p_nn_t;

void init_nn(nn_t *nn, int n_layers, int *layers_size);
#ifdef CPU
void train(nn_t *nn, ds_t *ds, int epochs, int batches, double lr);
void test(nn_t *nn, ds_t *ds);
#elif defined(GPU)
void train_GPU(nn_t *nn, ds_t *ds, int epochs, int batches, double lr);
void test_GPU(nn_t *nn, ds_t *ds);
#endif
void import_nn(nn_t *nn, char *filename);
void export_nn(nn_t *nn, char *filename);
void print_nn(nn_t *nn);
void print_deltas(nn_t *nn);

#endif
