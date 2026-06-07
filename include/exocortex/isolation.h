#ifndef EXOCORTEX_ISOLATION_H
#define EXOCORTEX_ISOLATION_H

/**
 * exocortex-kernel-c — Isolation Forest
 *
 * Anomaly detection via random isolation trees.
 * Tree traversal is ITERATIVE (explicit stack) — no recursion.
 *
 * Reference: Liu, Ting & Zhou, "Isolation Forest", ICDM 2008.
 */

typedef struct IsolationTree {
    int    split_feature;
    double split_value;
    int    is_leaf;
    int    size;             /* samples at this node */
    struct IsolationTree *left;
    struct IsolationTree *right;
} IsolationTree;

typedef struct {
    int             n_trees;
    int             sample_size;
    int             max_depth;
    IsolationTree  *trees;      /* array of n_trees root nodes */
    int             n_features; /* set during fit */
    int             n_training; /* set during fit, for normalization */
} IsolationForest;

/* Lifecycle */
int  isolation_forest_create(IsolationForest *forest, int n_trees, int sample_size);
void isolation_forest_free(IsolationForest *forest);

/* Fit — X: (n_samples × n_features) row-major */
int isolation_forest_fit(IsolationForest *forest, const double *X,
                         int n_samples, int n_features);

/* Score — returns anomaly score in [0,1]; higher = more anomalous */
double isolation_forest_score(const IsolationForest *forest, const double *x);

#endif /* EXOCORTEX_ISOLATION_H */
