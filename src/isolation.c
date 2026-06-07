/**
 * isolation.c — Isolation Forest implementation
 *
 * Iterative tree building and scoring (explicit stack, no recursion).
 * Reference: Liu, Ting & Zhou, ICDM 2008.
 */
#include "exocortex/isolation.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* Euler-Mascheroni constant for expected path length */
#define EULER_MASCHERONI 0.5772156649015329

static double expected_path_length(int n) {
    if (n <= 1) return 0.0;
    if (n == 2) return 1.0;
    double h = log((double)n) + EULER_MASCHERONI;
    return 2.0 * h - 2.0 * ((double)n - 1.0) / (double)n;
}

/* Stack node for iterative tree building */
typedef struct {
    IsolationTree *node;
    int           *indices;
    int            n_indices;
    int            depth;
} BuildFrame;

/* Stack node for iterative scoring */
typedef struct {
    const IsolationTree *node;
    int                  depth;
} ScoreFrame;

static void free_tree(IsolationTree *root) {
    if (!root) return;
    if (root->left) free_tree(root->left);
    if (root->right) free_tree(root->right);
    free(root);
}

int isolation_forest_create(IsolationForest *forest, int n_trees, int sample_size) {
    if (!forest || n_trees <= 0 || sample_size <= 0) return -1;
    forest->n_trees = n_trees;
    forest->sample_size = sample_size;
    forest->max_depth = (int)ceil(log2((double)sample_size));
    if (forest->max_depth < 1) forest->max_depth = 1;
    forest->trees = (IsolationTree *)calloc(n_trees, sizeof(IsolationTree));
    if (!forest->trees) return -1;
    memset(forest->trees, 0, n_trees * sizeof(IsolationTree));
    forest->n_features = 0;
    forest->n_training = 0;
    return 0;
}

void isolation_forest_free(IsolationForest *forest) {
    if (!forest) return;
    if (forest->trees) {
        for (int i = 0; i < forest->n_trees; i++) {
            if (forest->trees[i].left) free_tree(forest->trees[i].left);
            if (forest->trees[i].right) free_tree(forest->trees[i].right);
        }
        free(forest->trees);
    }
    forest->trees = NULL;
}

/* Build a single isolation tree iteratively */
static int build_tree(IsolationTree *root, const double *X, int *indices,
                      int n_indices, int n_features, int max_depth) {
    int frame_cap = 256;
    BuildFrame *stack = (BuildFrame *)malloc(frame_cap * sizeof(BuildFrame));
    if (!stack) return -1;
    int top = 0;

    root->split_feature = -1;
    root->split_value = 0.0;
    root->left = NULL;
    root->right = NULL;
    root->is_leaf = 0;
    root->size = n_indices;

    stack[top].node = root;
    stack[top].indices = indices;
    stack[top].n_indices = n_indices;
    stack[top].depth = 0;
    top++;

    while (top > 0) {
        top--;
        BuildFrame frame = stack[top];
        IsolationTree *node = frame.node;
        int n = frame.n_indices;
        int depth = frame.depth;

        if (n <= 1 || depth >= max_depth) {
            node->is_leaf = 1;
            node->size = n;
            node->left = NULL;
            node->right = NULL;
            continue;
        }

        /* Pick random feature */
        int feat = rand() % n_features;

        /* Find min/max for that feature */
        double fmin = DBL_MAX, fmax = -DBL_MAX;
        for (int i = 0; i < n; i++) {
            double val = X[indices[i] * n_features + feat];
            if (val < fmin) fmin = val;
            if (val > fmax) fmax = val;
        }

        if (fmin == fmax) {
            node->is_leaf = 1;
            node->size = n;
            node->left = NULL;
            node->right = NULL;
            continue;
        }

        /* Random split value in [fmin, fmax) */
        double r = (double)rand() / (RAND_MAX + 1.0);
        double split_val = fmin + r * (fmax - fmin);

        node->split_feature = feat;
        node->split_value = split_val;
        node->is_leaf = 0;
        node->size = n;

        /* Partition indices */
        int *left_idx = (int *)malloc(n * sizeof(int));
        int *right_idx = (int *)malloc(n * sizeof(int));
        if (!left_idx || !right_idx) { free(left_idx); free(right_idx); free(stack); return -1; }

        int nl = 0, nr = 0;
        for (int i = 0; i < n; i++) {
            double val = X[frame.indices[i] * n_features + feat];
            if (val < split_val)
                left_idx[nl++] = frame.indices[i];
            else
                right_idx[nr++] = frame.indices[i];
        }

        node->left = (IsolationTree *)calloc(1, sizeof(IsolationTree));
        node->right = (IsolationTree *)calloc(1, sizeof(IsolationTree));
        if (!node->left || !node->right) {
            free(left_idx); free(right_idx); free(stack);
            return -1;
        }

        /* Grow stack if needed */
        if (top + 2 > frame_cap) {
            frame_cap *= 2;
            stack = (BuildFrame *)realloc(stack, frame_cap * sizeof(BuildFrame));
            if (!stack) { free(left_idx); free(right_idx); return -1; }
        }

        stack[top].node = node->left;
        stack[top].indices = left_idx;
        stack[top].n_indices = nl;
        stack[top].depth = depth + 1;
        top++;

        stack[top].node = node->right;
        stack[top].indices = right_idx;
        stack[top].n_indices = nr;
        stack[top].depth = depth + 1;
        top++;
    }

    free(stack);
    return 0;
}

int isolation_forest_fit(IsolationForest *forest, const double *X,
                         int n_samples, int n_features) {
    if (!forest || !X || n_samples <= 0 || n_features <= 0) return -1;

    forest->n_features = n_features;
    forest->n_training = n_samples;

    int *all_indices = (int *)malloc(n_samples * sizeof(int));
    if (!all_indices) return -1;

    for (int t = 0; t < forest->n_trees; t++) {
        /* Subsample */
        int ss = forest->sample_size < n_samples ? forest->sample_size : n_samples;
        for (int i = 0; i < ss; i++)
            all_indices[i] = rand() % n_samples;

        build_tree(&forest->trees[t], X, all_indices, ss, n_features, forest->max_depth);
    }

    free(all_indices);
    return 0;
}

double isolation_forest_score(const IsolationForest *forest, const double *x) {
    if (!forest || !x) return -1.0;

    double total_path = 0.0;

    for (int t = 0; t < forest->n_trees; t++) {
        /* Iterative path length computation */
        int stack_cap = 64;
        ScoreFrame *stack = (ScoreFrame *)malloc(stack_cap * sizeof(ScoreFrame));
        if (!stack) return -1.0;
        int top = 0;

        stack[top].node = &forest->trees[t];
        stack[top].depth = 0;
        top++;

        double path_len = 0.0;
        while (top > 0) {
            top--;
            ScoreFrame frame = stack[top];
            const IsolationTree *node = frame.node;
            int depth = frame.depth;

            if (node->is_leaf || (!node->left && !node->right)) {
                path_len = (double)depth + expected_path_length(node->size);
                break;
            }

            if (x[node->split_feature] < node->split_value) {
                stack[top].node = node->left;
                stack[top].depth = depth + 1;
            } else {
                stack[top].node = node->right;
                stack[top].depth = depth + 1;
            }
            top++;
        }

        free(stack);
        total_path += path_len;
    }

    double avg_path = total_path / forest->n_trees;
    double c_n = expected_path_length(forest->n_training);
    if (c_n < 1e-12) c_n = 1.0;

    double score = pow(2.0, -avg_path / c_n);
    return score;
}
