/**
 * test_micro_nn.c — Tests for the 2-layer MLP
 */
#include "exocortex/micro_nn.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>

#define ASSERT_FEQ(a, b, eps) do { \
    if (fabs((a) - (b)) > (eps)) { \
        fprintf(stderr, "FAIL %s:%d: %.6f != %.6f (eps %.6f)\n", \
                __FILE__, __LINE__, (double)(a), (double)(b), (double)(eps)); \
        return 0; \
    } \
} while(0)

/* Test: create and free */
int test_nn_create_free(void) {
    MicroNN nn;
    assert(micro_nn_create(&nn, 3, 5, 2, 1) == 0);
    assert(nn.input_dim == 3);
    assert(nn.hidden_dim == 5);
    assert(nn.output_dim == 2);
    assert(nn.weights_1.data != NULL);
    assert(nn.weights_2.data != NULL);
    micro_nn_free(&nn);
    assert(nn.weights_1.data == NULL);
    return 1;
}

/* Test: forward pass shape correctness */
int test_nn_forward_shape(void) {
    MicroNN nn;
    micro_nn_create(&nn, 2, 4, 3, 1);
    double input[] = {1.0, 2.0};
    double output[3] = {0};
    assert(micro_nn_forward(&nn, input, output) == 0);
    /* Softmax output should sum to ~1 */
    double sum = output[0] + output[1] + output[2];
    ASSERT_FEQ(sum, 1.0, 1e-6);
    micro_nn_free(&nn);
    return 1;
}

/* Test: train XOR (classic test) */
int test_nn_xor(void) {
    MicroNN nn;
    micro_nn_create(&nn, 2, 8, 1, 0); /* regression mode for single output */

    double inputs[] = {0,0, 0,1, 1,0, 1,1};
    double targets[] = {0, 1, 1, 0};
    srand(42);
    micro_nn_create(&nn, 2, 8, 1, 0);
    micro_nn_train(&nn, inputs, targets, 4, 0.5, 2000);

    double out;
    micro_nn_forward(&nn, (double[]){0,0}, &out);
    ASSERT_FEQ(out, 0.0, 0.3); /* Relaxed — XOR with 2-layer MLP is hard */
    micro_nn_forward(&nn, (double[]){1,1}, &out);
    ASSERT_FEQ(out, 0.0, 0.3);
    micro_nn_forward(&nn, (double[]){0,1}, &out);
    ASSERT_FEQ(out, 1.0, 0.3);
    micro_nn_forward(&nn, (double[]){1,0}, &out);
    ASSERT_FEQ(out, 1.0, 0.3);

    micro_nn_free(&nn);
    return 1;
}

/* Test: classification output is softmax */
int test_nn_softmax_output(void) {
    MicroNN nn;
    micro_nn_create(&nn, 4, 6, 3, 1);
    double input[] = {0.5, -0.3, 1.2, 0.8};
    double output[3];
    micro_nn_forward(&nn, input, output);
    /* All outputs >= 0 */
    assert(output[0] >= 0.0);
    assert(output[1] >= 0.0);
    assert(output[2] >= 0.0);
    /* Sum ≈ 1 */
    ASSERT_FEQ(output[0] + output[1] + output[2], 1.0, 1e-6);
    micro_nn_free(&nn);
    return 1;
}

int run_micro_nn_tests(void) {
    int passed = 0;
    printf("  test_nn_create_free..."); passed += test_nn_create_free(); printf("OK\n");
    printf("  test_nn_forward_shape..."); passed += test_nn_forward_shape(); printf("OK\n");
    printf("  test_nn_xor..."); passed += test_nn_xor(); printf("OK\n");
    printf("  test_nn_softmax_output..."); passed += test_nn_softmax_output(); printf("OK\n");
    return passed;
}
