#ifndef EXOCORTEX_STATS_H
#define EXOCORTEX_STATS_H

/**
 * exocortex-kernel-c — Basic statistics
 *
 * mean, variance, standardize, argmax.
 * All operate on raw double arrays.
 */

double mean(const double *v, int n);
double variance(const double *v, int n);
int    standardize(double *v, int n, double *out);  /* zero mean, unit variance → out */
int    argmax(const double *v, int n);              /* index of max element */

#endif /* EXOCORTEX_STATS_H */
