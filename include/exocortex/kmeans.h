#ifndef EXOCORTEX_KMEANS_H
#define EXOCORTEX_KMEANS_H

/**
 * exocortex-kernel-c — K-Means Clustering
 *
 * Lloyd's algorithm with K-means++ initialization.
 */

typedef struct {
    int    k;            /* number of clusters */
    int    n_features;
    double *centroids;   /* k × n_features, row-major */
    int    *assignments; /* n_samples (set after fit) */
    int    n_assigned;   /* how many samples were fitted */
} KMeans;

/* Lifecycle */
int  kmeans_create(KMeans *km, int k, int n_features);
void kmeans_free(KMeans *km);

/* Fit — X: (n_samples × n_features) row-major */
int kmeans_fit(KMeans *km, const double *X, int n_samples, int max_iter);

/* Predict — returns cluster id [0, k) */
int kmeans_predict(const KMeans *km, const double *x);

#endif /* EXOCORTEX_KMEANS_H */
