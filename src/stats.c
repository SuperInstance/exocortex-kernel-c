/**
 * stats.c — Basic statistics implementation
 */
#include "exocortex/stats.h"
#include <math.h>

double mean(const double *v, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += v[i];
    return sum / n;
}

double variance(const double *v, int n) {
    double m = mean(v, n);
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double d = v[i] - m;
        sum += d * d;
    }
    return sum / n;
}

int standardize(double *v, int n, double *out) {
    if (n <= 0 || !out) return -1;
    double m = mean(v, n);
    double var = variance(v, n);
    double sd = sqrt(var);
    if (sd < 1e-12) {
        /* Constant vector → zero output */
        for (int i = 0; i < n; i++) out[i] = 0.0;
        return 0;
    }
    for (int i = 0; i < n; i++)
        out[i] = (v[i] - m) / sd;
    return 0;
}

int argmax(const double *v, int n) {
    if (n <= 0) return -1;
    int best = 0;
    for (int i = 1; i < n; i++)
        if (v[i] > v[best]) best = i;
    return best;
}
