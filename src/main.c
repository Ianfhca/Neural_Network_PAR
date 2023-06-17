#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ds.h"
#include "nn.cuh"
#include "utils.h"
#include "globals.h"

int main(int argc, char **argv) {

    ds_t ds;
    nn_t nn;
    
    if(argc == 1){
        printf("No arguments passed!\n");
        exit(0);
    }

    parse_arguments(argc, argv);

    if(train_mode){
        srand(seed);
        read_csv(dataset, &ds, layers[0], layers[n_layers - 1]);
        init_nn(&nn, n_layers, layers);
        #ifdef CPU
        printf("Comienza el entrenamiento con la CPU\n");
        train(&nn, &ds, epochs, batches, lr);
        #elif defined(GPU)
        printf("Comienza el entrenamiento con la GPU\n");
        train_GPU(&nn, &ds, epochs, batches, lr);
        #endif
        export_nn(&nn, model);
    }
    else if(test_mode){
        import_nn(&nn, model);
        read_csv(dataset, &ds, nn.layers_size[0], nn.layers_size[n_layers - 1]);
        #ifdef CPU
        test(&nn, &ds);
        #elif defined(GPU)

        #endif
    }
    
    return(0);
}

