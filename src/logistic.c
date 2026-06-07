/**
 * logistic.c — Logistic regression implementation
 *
 * Binary classification. Sigmoid activation, gradient descent.
 */
#include "exocortex/logistic.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static double sigmoid(double z) {
    if (z > 500.0) return 1.0;
    if (z < -500.0) return 0.0;
    return 1.0 / (1.0 + exp(-z));
}

int logistic_create(LogisticRegression *lr, int n_features) {
    if (!lr || n_features <= 0) return -1;
    lr->n_features = n_features;
    lr->weights = (double *)calloc(n_features, sizeof(double));
    lr->bias = 0.0;
    if (!lr->weights) return -1;

    /* Small random init */
    for (int i = 0; i < n_features; i++)
        lr->weights[i] = ((double)rand() / RAND_MAX - 0.5) * 0.1;

    return 0;
}

void logistic_free(LogisticRegression *lr) {
    if (lr && lr->weights) { free(lr->weights); lr->weights = NULL; }
}

int logistic_train(LogisticRegression *lr,
                   const double *X, const int *y,
                   int n_samples, double learning_rate, int epochs) {
    if (!lr || !X || !y || n_samples <= 0) return -1;

    int nf = lr->n_features;
    double *dw = (double *)calloc(nf, sizeof(double));
    if (!dw) return -1;

    for (int ep = 0; ep < epochs; ep++) {
        double db = 0.0;
        memset(dw, 0, nf * sizeof(double));

        for (int s = 0; s < n_samples; s++) {
            const double *x = X + s * nf;
            double z = lr->bias;
            for (int j = 0; j < nf; j++)
                z += lr->weights[j] * x[j];
            double p = sigmoid(z);
            double err = p - y[s];

            for (int j = 0; j < nf; j++)
                dw[j] += err * x[j];
            db += err;
        }

        double scale = learning_rate / n_samples;
        for (int j = 0; j < nf; j++)
            lr->weights[j] -= scale * dw[j];
        lr->bias -= scale * db;
    }

    free(dw);
    return 0;
}

int logistic_predict(const LogisticRegression *lr, const double *x, double *prob) {
    if (!lr || !x || !prob) return -1;
    double z = lr->bias;
    for (int j = 0; j < lr->n_features; j++)
        z += lr->weights[j] * x[j];
    *prob = sigmoid(z);
    return 0;
}
