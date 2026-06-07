#ifndef EXOCORTEX_COMPUTE_H
#define EXOCORTEX_COMPUTE_H

/**
 * exocortex-kernel-c — Unified Compute Kernel Interface
 *
 * A single struct that owns all model types and provides
 * a uniform create → train → predict lifecycle.
 *
 * model_type: "nn", "logistic", "kmeans", "isolation"
 */

#include "exocortex/micro_nn.h"
#include "exocortex/logistic.h"
#include "exocortex/kmeans.h"
#include "exocortex/isolation.h"

typedef enum {
    COMPUTE_MODEL_NN = 0,
    COMPUTE_MODEL_LOGISTIC,
    COMPUTE_MODEL_KMEANS,
    COMPUTE_MODEL_ISOLATION
} ComputeModelType;

typedef struct {
    ComputeModelType type;
    union {
        MicroNN           nn;
        LogisticRegression logistic;
        KMeans             kmeans;
        IsolationForest    isolation;
    } model;
} ComputeKernel;

/* Lifecycle */
int  compute_create(ComputeKernel *kernel, ComputeModelType type);
void compute_free(ComputeKernel *kernel);

#endif /* EXOCORTEX_COMPUTE_H */
