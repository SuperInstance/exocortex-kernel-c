#ifndef EXOCORTEX_MICRO_NN_H
#define EXOCORTEX_MICRO_NN_H

#include "exocortex/matrix.h"

/**
 * exocortex-kernel-c — 2-layer MLP (Micro Neural Network)
 *
 * Architecture: input → hidden (ReLU) → output (softmax / linear)
 * Training: SGD with full backpropagation, iteratively.
 *
 * Weights stored row-major in Matrix structs.
 *   weights_1: (input_dim × hidden)
 *   weights_2: (hidden × output_dim)
 */

typedef struct {
    int input_dim;
    int hidden_dim;
    int output_dim;
    int classification;   /* 1 = softmax output, 0 = linear output */
    Matrix weights_1;
    Matrix bias_1;        /* (1 × hidden) */
    Matrix weights_2;
    Matrix bias_2;        /* (1 × output_dim) */
} MicroNN;

/* Lifecycle */
int micro_nn_create(MicroNN *nn, int input_dim, int hidden, int output_dim, int classification);
void micro_nn_free(MicroNN *nn);

/* Inference */
int micro_nn_forward(const MicroNN *nn, const double *input, double *output);

/* Training — inputs: (n × input_dim) flat, targets: (n × output_dim) flat */
int micro_nn_train(MicroNN *nn,
                   const double *inputs, const double *targets,
                   int n_samples, double learning_rate, int epochs);

#endif /* EXOCORTEX_MICRO_NN_H */
