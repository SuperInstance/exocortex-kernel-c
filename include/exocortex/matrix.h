#ifndef EXOCORTEX_MATRIX_H
#define EXOCORTEX_MATRIX_H

/**
 * exocortex-kernel-c — Matrix operations
 *
 * Row-major dense matrix over double. No external deps.
 * Every function returns 0 on success, -1 on error.
 */

typedef struct {
    double *data;
    int rows;
    int cols;
} Matrix;

/* Lifecycle */
Matrix mat_create(int rows, int cols);           /* Returns zero-filled matrix; check .data != NULL */
void   mat_free(Matrix *m);

/* Accessors */
static inline double mat_get(const Matrix *m, int r, int c) {
    return m->data[r * m->cols + c];
}
static inline void mat_set(Matrix *m, int r, int c, double val) {
    m->data[r * m->cols + c] = val;
}

/* Operations — out must be pre-allocated; returns 0/-1 */
int mat_mul(const Matrix *a, const Matrix *b, Matrix *out);       /* out = a × b */
int mat_transpose(const Matrix *a, Matrix *out);                   /* out = aᵀ */
int mat_add(const Matrix *a, const Matrix *b, Matrix *out);       /* out = a + b */
int mat_scale(Matrix *m, double scalar);                           /* m *= scalar */
int mat_apply(Matrix *m, double (*fn)(double));                    /* element-wise */

/* Utility */
void mat_print(const Matrix *m, const char *label);

#endif /* EXOCORTEX_MATRIX_H */
