/**
 * compute.c — Unified Compute Kernel Interface
 */
#include "exocortex/compute.h"
#include <stdlib.h>
#include <string.h>

int compute_create(ComputeKernel *kernel, ComputeModelType type) {
    if (!kernel) return -1;
    memset(kernel, 0, sizeof(ComputeKernel));
    kernel->type = type;

    switch (type) {
        case COMPUTE_MODEL_NN:
            /* Caller must call micro_nn_create separately with dims */
            return 0;
        case COMPUTE_MODEL_LOGISTIC:
            return 0;
        case COMPUTE_MODEL_KMEANS:
            return 0;
        case COMPUTE_MODEL_ISOLATION:
            return 0;
        default:
            return -1;
    }
}

void compute_free(ComputeKernel *kernel) {
    if (!kernel) return;
    switch (kernel->type) {
        case COMPUTE_MODEL_NN:
            micro_nn_free(&kernel->model.nn);
            break;
        case COMPUTE_MODEL_LOGISTIC:
            logistic_free(&kernel->model.logistic);
            break;
        case COMPUTE_MODEL_KMEANS:
            kmeans_free(&kernel->model.kmeans);
            break;
        case COMPUTE_MODEL_ISOLATION:
            isolation_forest_free(&kernel->model.isolation);
            break;
    }
}
