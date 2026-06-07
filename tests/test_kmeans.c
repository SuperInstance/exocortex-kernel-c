/**
 * test_kmeans.c — Tests for K-Means clustering
 */
#include "exocortex/kmeans.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>

/* Test: create/free */
int test_kmeans_create_free(void) {
    KMeans km;
    assert(kmeans_create(&km, 3, 2) == 0);
    assert(km.k == 3);
    assert(km.n_features == 2);
    assert(km.centroids != NULL);
    kmeans_free(&km);
    assert(km.centroids == NULL);
    return 1;
}

/* Test: fit finds two clusters */
int test_kmeans_two_clusters(void) {
    KMeans km;
    srand(42);
    kmeans_create(&km, 2, 2);

    /* Two clear clusters */
    double X[] = {
        -5.0, -5.0,
        -4.5, -5.5,
        -5.5, -4.5,
         5.0,  5.0,
         4.5,  5.5,
         5.5,  4.5
    };

    assert(kmeans_fit(&km, X, 6, 100) == 0);
    assert(km.assignments != NULL);

    /* Points in same cluster should have same assignment */
    assert(km.assignments[0] == km.assignments[1]);
    assert(km.assignments[0] == km.assignments[2]);
    assert(km.assignments[3] == km.assignments[4]);
    assert(km.assignments[3] == km.assignments[5]);

    /* Different clusters */
    assert(km.assignments[0] != km.assignments[3]);

    kmeans_free(&km);
    return 1;
}

/* Test: predict assigns to nearest centroid */
int test_kmeans_predict(void) {
    KMeans km;
    srand(42);
    kmeans_create(&km, 2, 2);

    double X[] = { -10,-10, 10,10 };
    kmeans_fit(&km, X, 2, 10);

    int c1 = kmeans_predict(&km, (double[]){-10,-10});
    int c2 = kmeans_predict(&km, (double[]){10,10});
    assert(c1 != c2);

    kmeans_free(&km);
    return 1;
}

/* Test: three clusters */
int test_kmeans_three_clusters(void) {
    KMeans km;
    srand(42);
    kmeans_create(&km, 3, 2);

    double X[] = {
        -10, 0,  -9, 1,  -10, -1,
         10, 0,   9, 1,   10, -1,
         0, 10,   1, 9,  -1, 10
    };

    kmeans_fit(&km, X, 9, 100);

    /* Each group of 3 should be in the same cluster */
    assert(km.assignments[0] == km.assignments[1]);
    assert(km.assignments[0] == km.assignments[2]);
    assert(km.assignments[3] == km.assignments[4]);
    assert(km.assignments[3] == km.assignments[5]);
    assert(km.assignments[6] == km.assignments[7]);
    assert(km.assignments[6] == km.assignments[8]);

    /* All 3 groups in different clusters */
    assert(km.assignments[0] != km.assignments[3]);
    assert(km.assignments[0] != km.assignments[6]);
    assert(km.assignments[3] != km.assignments[6]);

    kmeans_free(&km);
    return 1;
}

int run_kmeans_tests(void) {
    int passed = 0;
    printf("  test_kmeans_create_free..."); passed += test_kmeans_create_free(); printf("OK\n");
    printf("  test_kmeans_two_clusters..."); passed += test_kmeans_two_clusters(); printf("OK\n");
    printf("  test_kmeans_predict..."); passed += test_kmeans_predict(); printf("OK\n");
    printf("  test_kmeans_three_clusters..."); passed += test_kmeans_three_clusters(); printf("OK\n");
    return passed;
}
