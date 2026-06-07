/**
 * test_logistic.c — Tests for logistic regression
 */
#include "exocortex/logistic.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#define ASSERT_FEQ(a, b, eps) do { \
    if (fabs((a) - (b)) > (eps)) { \
        fprintf(stderr, "FAIL %s:%d: %.6f != %.6f\n", \
                __FILE__, __LINE__, (double)(a), (double)(b)); \
        return 0; \
    } \
} while(0)

/* Test: create/free lifecycle */
int test_logistic_create_free(void) {
    LogisticRegression lr;
    assert(logistic_create(&lr, 5) == 0);
    assert(lr.n_features == 5);
    assert(lr.weights != NULL);
    logistic_free(&lr);
    assert(lr.weights == NULL);
    return 1;
}

/* Test: AND gate classification */
int test_logistic_and(void) {
    LogisticRegression lr;
    srand(42);
    logistic_create(&lr, 2);

    double X[] = {0,0, 0,1, 1,0, 1,1};
    int    y[] = {0,   0,   0,   1};
    logistic_train(&lr, X, y, 4, 2.0, 500);

    double prob;
    logistic_predict(&lr, (double[]){0,0}, &prob);
    assert(prob < 0.5);

    logistic_predict(&lr, (double[]){1,1}, &prob);
    assert(prob > 0.5);

    logistic_free(&lr);
    return 1;
}

/* Test: sigmoid bounds */
int test_logistic_bounds(void) {
    LogisticRegression lr;
    srand(42);
    logistic_create(&lr, 2);

    /* With zero weights and zero bias, sigmoid(0) = 0.5 */
    memset(lr.weights, 0, 2 * sizeof(double));
    lr.bias = 0.0;
    double prob;
    logistic_predict(&lr, (double[]){0,0}, &prob);
    ASSERT_FEQ(prob, 0.5, 1e-6);

    logistic_free(&lr);
    return 1;
}

/* Test: predict on trained model separates classes */
int test_logistic_separation(void) {
    LogisticRegression lr;
    srand(42);
    logistic_create(&lr, 2);

    /* Two well-separated clusters */
    double X[] = {-5,-5, -4,-6, -5,-4,   5,5, 6,4, 4,6};
    int    y[] = {0, 0, 0,           1, 1, 1};
    logistic_train(&lr, X, y, 6, 1.0, 200);

    double p0, p1;
    logistic_predict(&lr, (double[]){-5,-5}, &p0);
    logistic_predict(&lr, (double[]){5,5}, &p1);
    assert(p0 < p1);
    assert(p0 < 0.5);
    assert(p1 > 0.5);

    logistic_free(&lr);
    return 1;
}

int run_logistic_tests(void) {
    int passed = 0;
    printf("  test_logistic_create_free..."); passed += test_logistic_create_free(); printf("OK\n");
    printf("  test_logistic_and..."); passed += test_logistic_and(); printf("OK\n");
    printf("  test_logistic_bounds..."); passed += test_logistic_bounds(); printf("OK\n");
    printf("  test_logistic_separation..."); passed += test_logistic_separation(); printf("OK\n");
    return passed;
}
