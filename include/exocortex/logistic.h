#ifndef EXOCORTEX_LOGISTIC_H
#define EXOCORTEX_LOGISTIC_H

/**
 * exocortex-kernel-c — Logistic Regression
 *
 * Binary classification via sigmoid + gradient descent.
 */

typedef struct {
    int    n_features;
    double *weights;    /* length n_features */
    double bias;
} LogisticRegression;

/* Lifecycle */
int  logistic_create(LogisticRegression *lr, int n_features);
void logistic_free(LogisticRegression *lr);

/* Training — X: (n_samples × n_features) row-major, y: binary labels */
int logistic_train(LogisticRegression *lr,
                   const double *X, const int *y,
                   int n_samples, double learning_rate, int epochs);

/* Predict — returns probability in *prob; classify as >= 0.5 */
int logistic_predict(const LogisticRegression *lr, const double *x, double *prob);

#endif /* EXOCORTEX_LOGISTIC_H */
