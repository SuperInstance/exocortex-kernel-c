/**
 * micro_nn.c — 2-layer MLP implementation
 *
 * Forward: z1 = input × W1 + b1, a1 = ReLU(z1), z2 = a1 × W2 + b2, output = softmax/linear(z2)
 * Backprop: compute gradients, update weights via SGD.
 */
#include "exocortex/micro_nn.h"
#include "exocortex/stats.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

static double relu(double x) { return x > 0.0 ? x : 0.0; }
static double relu_deriv(double x) { return x > 0.0 ? 1.0 : 0.0; }

static void softmax(double *v, int n) {
    double max_v = v[0];
    for (int i = 1; i < n; i++) if (v[i] > max_v) max_v = v[i];
    double sum = 0.0;
    for (int i = 0; i < n; i++) { v[i] = exp(v[i] - max_v); sum += v[i]; }
    for (int i = 0; i < n; i++) v[i] /= sum;
}

/* Xavier-ish random init */
static void randomize_matrix(Matrix *m) {
    int total = m->rows * m->cols;
    double scale = total > 0 ? sqrt(2.0 / total) : 1.0;
    for (int i = 0; i < total; i++) {
        double u1 = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
        double u2 = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
        m->data[i] = scale * sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
    }
}

int micro_nn_create(MicroNN *nn, int input_dim, int hidden, int output_dim, int classification) {
    if (!nn || input_dim <= 0 || hidden <= 0 || output_dim <= 0) return -1;
    nn->input_dim = input_dim;
    nn->hidden_dim = hidden;
    nn->output_dim = output_dim;
    nn->classification = classification;

    nn->weights_1 = mat_create(input_dim, hidden);
    nn->bias_1    = mat_create(1, hidden);
    nn->weights_2 = mat_create(hidden, output_dim);
    nn->bias_2    = mat_create(1, output_dim);

    if (!nn->weights_1.data || !nn->bias_1.data ||
        !nn->weights_2.data || !nn->bias_2.data) {
        micro_nn_free(nn);
        return -1;
    }

    randomize_matrix(&nn->weights_1);
    randomize_matrix(&nn->weights_2);
    /* Biases stay zero-initialized from calloc */

    return 0;
}

void micro_nn_free(MicroNN *nn) {
    if (!nn) return;
    mat_free(&nn->weights_1);
    mat_free(&nn->bias_1);
    mat_free(&nn->weights_2);
    mat_free(&nn->bias_2);
}

int micro_nn_forward(const MicroNN *nn, const double *input, double *output) {
    if (!nn || !input || !output) return -1;

    /* Hidden layer: z1 = input(1×in) × W1(in×hid) + b1(1×hid) */
    double *z1 = (double *)calloc(nn->hidden_dim, sizeof(double));
    double *a1 = (double *)calloc(nn->hidden_dim, sizeof(double));
    if (!z1 || !a1) { free(z1); free(a1); return -1; }

    for (int h = 0; h < nn->hidden_dim; h++) {
        double sum = nn->bias_1.data[h];
        for (int i = 0; i < nn->input_dim; i++)
            sum += input[i] * mat_get(&nn->weights_1, i, h);
        z1[h] = sum;
        a1[h] = relu(sum);
    }

    /* Output layer: z2 = a1(1×hid) × W2(hid×out) + b2(1×out) */
    for (int o = 0; o < nn->output_dim; o++) {
        double sum = nn->bias_2.data[o];
        for (int h = 0; h < nn->hidden_dim; h++)
            sum += a1[h] * mat_get(&nn->weights_2, h, o);
        output[o] = sum;
    }

    if (nn->classification)
        softmax(output, nn->output_dim);

    free(z1);
    free(a1);
    return 0;
}

int micro_nn_train(MicroNN *nn,
                   const double *inputs, const double *targets,
                   int n_samples, double learning_rate, int epochs) {
    if (!nn || !inputs || !targets || n_samples <= 0) return -1;

    int in_d = nn->input_dim, hid = nn->hidden_dim, out_d = nn->output_dim;

    /* Allocate gradient buffers */
    double *dW1 = (double *)calloc(in_d * hid, sizeof(double));
    double *db1 = (double *)calloc(hid, sizeof(double));
    double *dW2 = (double *)calloc(hid * out_d, sizeof(double));
    double *db2 = (double *)calloc(out_d, sizeof(double));
    double *z1  = (double *)calloc(hid, sizeof(double));
    double *a1  = (double *)calloc(hid, sizeof(double));
    double *z2  = (double *)calloc(out_d, sizeof(double));
    double *dz2 = (double *)calloc(out_d, sizeof(double));
    double *da1 = (double *)calloc(hid, sizeof(double));

    if (!dW1 || !db1 || !dW2 || !db2 || !z1 || !a1 || !z2 || !dz2 || !da1) {
        free(dW1); free(db1); free(dW2); free(db2);
        free(z1); free(a1); free(z2); free(dz2); free(da1);
        return -1;
    }

    for (int ep = 0; ep < epochs; ep++) {
        /* Zero gradients */
        memset(dW1, 0, in_d * hid * sizeof(double));
        memset(db1, 0, hid * sizeof(double));
        memset(dW2, 0, hid * out_d * sizeof(double));
        memset(db2, 0, out_d * sizeof(double));

        for (int s = 0; s < n_samples; s++) {
            const double *x = inputs + s * in_d;
            const double *t = targets + s * out_d;

            /* Forward: hidden */
            for (int h = 0; h < hid; h++) {
                double sum = nn->bias_1.data[h];
                for (int i = 0; i < in_d; i++)
                    sum += x[i] * mat_get(&nn->weights_1, i, h);
                z1[h] = sum;
                a1[h] = relu(sum);
            }

            /* Forward: output */
            for (int o = 0; o < out_d; o++) {
                double sum = nn->bias_2.data[o];
                for (int h = 0; h < hid; h++)
                    sum += a1[h] * mat_get(&nn->weights_2, h, o);
                z2[o] = sum;
            }

            /* dz2 = output - target (cross-entropy + softmax simplification, or MSE) */
            if (nn->classification) {
                /* softmax */
                double *sm = z2;
                softmax(sm, out_d);
                for (int o = 0; o < out_d; o++)
                    dz2[o] = sm[o] - t[o];
            } else {
                for (int o = 0; o < out_d; o++)
                    dz2[o] = z2[o] - t[o];
            }

            /* Accumulate dW2, db2 */
            for (int h = 0; h < hid; h++)
                for (int o = 0; o < out_d; o++)
                    dW2[h * out_d + o] += a1[h] * dz2[o];
            for (int o = 0; o < out_d; o++)
                db2[o] += dz2[o];

            /* Backprop to hidden: da1 = dz2 × W2ᵀ */
            for (int h = 0; h < hid; h++) {
                da1[h] = 0.0;
                for (int o = 0; o < out_d; o++)
                    da1[h] += dz2[o] * mat_get(&nn->weights_2, h, o);
                da1[h] *= relu_deriv(z1[h]); /* dz1 */
            }

            /* Accumulate dW1, db1 */
            for (int i = 0; i < in_d; i++)
                for (int h = 0; h < hid; h++)
                    dW1[i * hid + h] += x[i] * da1[h];
            for (int h = 0; h < hid; h++)
                db1[h] += da1[h];
        }

        /* Apply gradients (SGD) */
        double scale = learning_rate / n_samples;
        for (int i = 0; i < in_d * hid; i++)
            nn->weights_1.data[i] -= scale * dW1[i];
        for (int h = 0; h < hid; h++)
            nn->bias_1.data[h] -= scale * db1[h];
        for (int i = 0; i < hid * out_d; i++)
            nn->weights_2.data[i] -= scale * dW2[i];
        for (int o = 0; o < out_d; o++)
            nn->bias_2.data[o] -= scale * db2[o];
    }

    free(dW1); free(db1); free(dW2); free(db2);
    free(z1); free(a1); free(z2); free(dz2); free(da1);
    return 0;
}
