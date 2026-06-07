/**
 * matrix.c — Matrix operations implementation
 */
#include "exocortex/matrix.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

Matrix mat_create(int rows, int cols) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.data = (double *)calloc((size_t)rows * cols, sizeof(double));
    return m;
}

void mat_free(Matrix *m) {
    if (m->data) { free(m->data); m->data = NULL; }
    m->rows = m->cols = 0;
}

int mat_mul(const Matrix *a, const Matrix *b, Matrix *out) {
    if (a->cols != b->rows || out->rows != a->rows || out->cols != b->cols)
        return -1;
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            double sum = 0.0;
            for (int k = 0; k < a->cols; k++) {
                sum += mat_get(a, i, k) * mat_get(b, k, j);
            }
            mat_set(out, i, j, sum);
        }
    }
    return 0;
}

int mat_transpose(const Matrix *a, Matrix *out) {
    if (out->rows != a->cols || out->cols != a->rows)
        return -1;
    for (int i = 0; i < a->rows; i++)
        for (int j = 0; j < a->cols; j++)
            mat_set(out, j, i, mat_get(a, i, j));
    return 0;
}

int mat_add(const Matrix *a, const Matrix *b, Matrix *out) {
    if (a->rows != b->rows || a->cols != b->cols ||
        out->rows != a->rows || out->cols != a->cols)
        return -1;
    for (int i = 0; i < a->rows * a->cols; i++)
        out->data[i] = a->data[i] + b->data[i];
    return 0;
}

int mat_scale(Matrix *m, double scalar) {
    if (!m->data) return -1;
    for (int i = 0; i < m->rows * m->cols; i++)
        m->data[i] *= scalar;
    return 0;
}

int mat_apply(Matrix *m, double (*fn)(double)) {
    if (!m->data || !fn) return -1;
    for (int i = 0; i < m->rows * m->cols; i++)
        m->data[i] = fn(m->data[i]);
    return 0;
}

void mat_print(const Matrix *m, const char *label) {
    printf("%s (%dx%d):\n", label, m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        printf("  [");
        for (int j = 0; j < m->cols; j++) {
            printf("%8.4f", mat_get(m, i, j));
            if (j < m->cols - 1) printf(", ");
        }
        printf("]\n");
    }
}
