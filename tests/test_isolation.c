/**
 * test_isolation.c — Tests for Isolation Forest
 */
#include "exocortex/isolation.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>

#define ASSERT_FEQ(a, b, eps) do { \
    if (fabs((a) - (b)) > (eps)) { \
        fprintf(stderr, "FAIL %s:%d: %.6f != %.6f\n", \
                __FILE__, __LINE__, (double)(a), (double)(b)); \
        return 0; \
    } \
} while(0)

/* Test: create/free */
int test_isolation_create_free(void) {
    IsolationForest forest;
    assert(isolation_forest_create(&forest, 10, 50) == 0);
    assert(forest.n_trees == 10);
    assert(forest.sample_size == 50);
    assert(forest.trees != NULL);
    isolation_forest_free(&forest);
    assert(forest.trees == NULL);
    return 1;
}

/* Test: anomaly score is in [0, 1] */
int test_isolation_score_range(void) {
    IsolationForest forest;
    srand(42);
    isolation_forest_create(&forest, 20, 5);

    /* Normal data clustered around origin */
    double X[] = {
        0.1, 0.2, 0.15, 0.18, 0.12,
       -0.1,-0.2,-0.15,-0.18,-0.12,
        0.2, 0.1, 0.18, 0.15, 0.11,
       -0.2,-0.1,-0.18,-0.15,-0.11,
        0.05, 0.08, 0.03, 0.07, 0.04
    };
    isolation_forest_fit(&forest, X, 5, 5);

    /* Score a normal point */
    double score_normal = isolation_forest_score(&forest, (double[]){0.1, 0.1, 0.1, 0.1, 0.1});
    assert(score_normal >= 0.0);
    assert(score_normal <= 1.0);

    /* Score an anomalous point */
    double score_anomaly = isolation_forest_score(&forest, (double[]){100.0, 100.0, 100.0, 100.0, 100.0});
    assert(score_anomaly >= 0.0);
    assert(score_anomaly <= 1.0);

    isolation_forest_free(&forest);
    return 1;
}

/* Test: anomalies score higher than normal points */
int test_isolation_anomaly_detection(void) {
    IsolationForest forest;
    srand(42);
    isolation_forest_create(&forest, 50, 20);

    /* Generate clustered normal data + fit */
    #define N_NORMAL 40
    double X[N_NORMAL * 2];
    for (int i = 0; i < N_NORMAL; i++) {
        X[i*2]     = ((double)rand() / RAND_MAX) * 2.0 - 1.0;  /* [-1, 1] */
        X[i*2 + 1] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }
    isolation_forest_fit(&forest, X, N_NORMAL, 2);

    double score_normal  = isolation_forest_score(&forest, (double[]){0.0, 0.0});
    double score_anomaly = isolation_forest_score(&forest, (double[]){50.0, 50.0});

    assert(score_anomaly > score_normal);

    isolation_forest_free(&forest);
    return 1;
}

/* Test: fit returns error on bad input */
int test_isolation_error_handling(void) {
    IsolationForest forest;
    assert(isolation_forest_create(NULL, 10, 5) == -1);
    assert(isolation_forest_create(&forest, 0, 5) == -1);
    return 1;
}

int run_isolation_tests(void) {
    int passed = 0;
    printf("  test_isolation_create_free..."); passed += test_isolation_create_free(); printf("OK\n");
    printf("  test_isolation_score_range..."); passed += test_isolation_score_range(); printf("OK\n");
    printf("  test_isolation_anomaly_detection..."); passed += test_isolation_anomaly_detection(); printf("OK\n");
    printf("  test_isolation_error_handling..."); passed += test_isolation_error_handling(); printf("OK\n");
    return passed;
}
