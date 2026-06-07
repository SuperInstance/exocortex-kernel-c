/**
 * test_main.c — Test runner for exocortex-kernel-c
 */
#include <stdio.h>
#include <stdlib.h>

extern int run_micro_nn_tests(void);
extern int run_logistic_tests(void);
extern int run_kmeans_tests(void);
extern int run_isolation_tests(void);

int main(void) {
    int total = 0, passed = 0;

    printf("=== MicroNN Tests ===\n");
    passed += run_micro_nn_tests();
    total += 4;

    printf("\n=== Logistic Regression Tests ===\n");
    passed += run_logistic_tests();
    total += 4;

    printf("\n=== K-Means Tests ===\n");
    passed += run_kmeans_tests();
    total += 4;

    printf("\n=== Isolation Forest Tests ===\n");
    passed += run_isolation_tests();
    total += 4;

    printf("\n========================================\n");
    printf("Results: %d / %d tests passed\n", passed, total);
    printf("========================================\n");

    return (passed == total) ? 0 : 1;
}
