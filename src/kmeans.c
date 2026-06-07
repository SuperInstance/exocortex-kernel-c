/**
 * kmeans.c — K-Means clustering implementation
 *
 * K-means++ initialization + Lloyd's algorithm.
 */
#include "exocortex/kmeans.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

int kmeans_create(KMeans *km, int k, int n_features) {
    if (!km || k <= 0 || n_features <= 0) return -1;
    km->k = k;
    km->n_features = n_features;
    km->centroids = (double *)calloc(k * n_features, sizeof(double));
    km->assignments = NULL;
    km->n_assigned = 0;
    if (!km->centroids) return -1;
    return 0;
}

void kmeans_free(KMeans *km) {
    if (!km) return;
    if (km->centroids) { free(km->centroids); km->centroids = NULL; }
    if (km->assignments) { free(km->assignments); km->assignments = NULL; }
}

static double dist_sq(const double *a, const double *b, int dim) {
    double d = 0.0;
    for (int i = 0; i < dim; i++) {
        double diff = a[i] - b[i];
        d += diff * diff;
    }
    return d;
}

int kmeans_fit(KMeans *km, const double *X, int n_samples, int max_iter) {
    if (!km || !X || n_samples <= 0) return -1;

    int k = km->k, nf = km->n_features;

    /* Allocate assignments */
    if (km->assignments) free(km->assignments);
    km->assignments = (int *)malloc(n_samples * sizeof(int));
    if (!km->assignments) return -1;
    km->n_assigned = n_samples;

    /* K-means++ initialization */
    /* Pick first centroid randomly */
    int first = rand() % n_samples;
    memcpy(km->centroids, X + first * nf, nf * sizeof(double));

    double *min_dists = (double *)malloc(n_samples * sizeof(double));
    if (!min_dists) return -1;

    for (int c = 1; c < k; c++) {
        /* Compute min distance to existing centroids */
        double total = 0.0;
        for (int s = 0; s < n_samples; s++) {
            double md = DBL_MAX;
            for (int pc = 0; pc < c; pc++) {
                double d = dist_sq(X + s * nf, km->centroids + pc * nf, nf);
                if (d < md) md = d;
            }
            min_dists[s] = md;
            total += md;
        }

        /* Weighted random selection */
        double r = ((double)rand() / RAND_MAX) * total;
        double cumulative = 0.0;
        int chosen = 0;
        for (int s = 0; s < n_samples; s++) {
            cumulative += min_dists[s];
            if (cumulative >= r) { chosen = s; break; }
        }
        memcpy(km->centroids + c * nf, X + chosen * nf, nf * sizeof(double));
    }

    free(min_dists);

    /* Lloyd's iterations */
    int *counts = (int *)calloc(k, sizeof(int));
    double *new_centroids = (double *)calloc(k * nf, sizeof(double));
    if (!counts || !new_centroids) {
        free(counts); free(new_centroids);
        return -1;
    }

    for (int iter = 0; iter < max_iter; iter++) {
        /* Assign each point to nearest centroid */
        int changed = 0;
        for (int s = 0; s < n_samples; s++) {
            double best_d = DBL_MAX;
            int best_c = 0;
            for (int c = 0; c < k; c++) {
                double d = dist_sq(X + s * nf, km->centroids + c * nf, nf);
                if (d < best_d) { best_d = d; best_c = c; }
            }
            if (km->assignments[s] != best_c) changed = 1;
            km->assignments[s] = best_c;
        }

        if (!changed && iter > 0) break;

        /* Recompute centroids */
        memset(new_centroids, 0, k * nf * sizeof(double));
        memset(counts, 0, k * sizeof(int));
        for (int s = 0; s < n_samples; s++) {
            int c = km->assignments[s];
            counts[c]++;
            for (int f = 0; f < nf; f++)
                new_centroids[c * nf + f] += X[s * nf + f];
        }
        for (int c = 0; c < k; c++) {
            if (counts[c] > 0) {
                for (int f = 0; f < nf; f++)
                    km->centroids[c * nf + f] = new_centroids[c * nf + f] / counts[c];
            }
        }
    }

    free(counts);
    free(new_centroids);
    return 0;
}

int kmeans_predict(const KMeans *km, const double *x) {
    if (!km || !x) return -1;
    double best_d = DBL_MAX;
    int best_c = 0;
    for (int c = 0; c < km->k; c++) {
        double d = dist_sq(x, km->centroids + c * km->n_features, km->n_features);
        if (d < best_d) { best_d = d; best_c = c; }
    }
    return best_c;
}
