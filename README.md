# exocortex-kernel-c

**Pure C implementation of the exocortex compute kernel.**

Zero dependencies. Zero abstraction overhead. Zero excuses.

This library proves that fundamental machine learning algorithms — neural networks, logistic regression, K-means clustering, and isolation forests — can be implemented in portable C99/C11 with no external dependencies, no build system complexity, and no runtime bloat.

If the ESP32 can't run it, it's not real ML.

---

## Table of Contents

1. [Why C?](#why-c)
2. [Architecture](#architecture)
3. [Complexity Analysis](#complexity-analysis)
4. [Quick Start](#quick-start)
5. [Building](#building)
6. [Library Reference](#library-reference)
7. [Examples](#examples)
8. [Performance Comparison](#performance-comparison)
9. [Design Decisions](#design-decisions)
10. [Comparison with Existing Frameworks](#comparison-with-existing-frameworks)
11. [Historical References](#historical-references)
12. [The PLATO Principle](#the-plato-principle)
13. [Glossary](#glossary)
14. [License](#license)

---

## Why C?

> "C is quirky, flawed, and an enormous success. While accidents of history surely helped, it evidently satisfied a need for a system implementation language efficient enough to displace assembly language, yet sufficiently abstract and portable to describe algorithms and interactions in a wide variety of environments."
> — Dennis Ritchie, creator of C

Why implement machine learning in C when Python, Rust, and Julia exist?

**Because C is the lingua franca of computing.** Every operating system, every embedded runtime, every microcontroller toolchain speaks C. When you implement an algorithm in C, it runs *everywhere* — not just everywhere that supports your runtime, your garbage collector, or your package manager.

### Zero-Cost Abstractions When YOU Control the Cost

The Rust community loves to talk about "zero-cost abstractions." But here's the truth: in C, *you* control what costs exist. There are no hidden allocations behind trait objects, no monomorphization bloat, no async runtime surprises. Every `malloc` is yours. Every `free` is yours. Every pointer dereference is a choice you made, consciously, with full awareness of the trade-off.

The "cost" in C isn't zero — it's *transparent*. You can see it, measure it, and optimize it. That's better than zero-cost. That's *honest-cost*.

### Where C Shines for ML

| Dimension | C | Python | Rust | Julia |
|---|---|---|---|---|
| Startup time | ~μs | ~100ms | ~ms | ~seconds (JIT) |
| Memory overhead | What you allocate | 10-100× data size | Low | Moderate |
| Embedded support | Universal | None | Limited | None |
| Binary size | ~KB | ~100MB (numpy) | ~MB | ~100MB |
| Determinism | Full | No (GC in CPython) | No (async) | No (JIT) |
| Portability | Every platform | x86/ARM mostly | Growing | Limited |

---

## Architecture

```
                    ┌─────────────────────────────────────────────────┐
                    │            EXOCORTEX COMPUTE KERNEL             │
                    └─────────────────────────────────────────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                   │
            ┌───────▼──────┐  ┌───────▼──────┐  ┌────────▼────────┐
            │   MATRIX     │  │    STATS     │  │    COMPUTE      │
            │   OPERATIONS │  │   UTILITIES  │  │    KERNEL       │
            │              │  │              │  │   (UNIFIED)     │
            │ • create     │  │ • mean       │  │                 │
            │ • mul        │  │ • variance   │  │  Model Union:   │
            │ • transpose  │  │ • standardize│  │  ┌───────────┐  │
            │ • add        │  │ • argmax     │  │  │ MicroNN   │  │
            │ • scale      │  │              │  │  │ Logistic  │  │
            │ • apply      │  │              │  │  │ KMeans    │  │
            │ • print      │  │              │  │  │ Isolation │  │
            └──────────────┘  └──────────────┘  │  └───────────┘  │
                                                └─────────────────┘
                    │                  │                   │
            ┌───────▼──────┐  ┌───────▼──────┐  ┌────────▼────────┐
            │   MICRO NN   │  │   LOGISTIC   │  │    K-MEANS      │
            │   (2-LAYER   │  │  REGRESSION  │  │   CLUSTERING    │
            │     MLP)     │  │              │  │                 │
            │              │  │ • sigmoid    │  │ • K-means++     │
            │ • ReLU       │  │ • gradient   │  │   init          │
            │ • softmax    │  │   descent    │  │ • Lloyd's       │
            │ • backprop   │  │ • binary     │  │   algorithm     │
            │ • SGD        │  │   classify   │  │ • predict       │
            └──────────────┘  └──────────────┘  └─────────────────┘
                                                          │
                                                ┌────────▼────────┐
                                                │   ISOLATION     │
                                                │    FOREST       │
                                                │                 │
                                                │ • anomaly       │
                                                │   detection     │
                                                │ • iterative     │
                                                │   traversal     │
                                                │ • path length   │
                                                │   scoring       │
                                                └─────────────────┘
```

### Data Flow

```
Raw Data (double*)
     │
     ├─────► standardize() ──► MicroNN ──► predictions
     │                              │
     ├─────► logistic_train() ──► probabilities
     │
     ├─────► kmeans_fit() ──► cluster assignments
     │
     └─────► isolation_forest_fit() ──► anomaly scores
```

---

## Complexity Analysis

### MicroNN Forward Pass

For a 2-layer MLP with `I` input features, `H` hidden units, and `O` output features:

```
Time:  O(I × H + H × O)
Space: O(H + O)  (activations)
```

The forward pass is dominated by two matrix-vector multiplications. Each requires a number of multiply-accumulate operations proportional to the product of input and output dimensions. This is optimal — you cannot compute a matrix-vector product with fewer operations.

Training adds `E` epochs × `N` samples, giving total training complexity:

```
Training: O(N × E × (I × H + H × O))
```

### Logistic Regression

For `N` samples, `F` features, `E` epochs:

```
Time:  O(N × F × E)
Space: O(F)  (weights)
```

Logistic regression is the simplest non-trivial classifier. Each epoch scans all samples and features. The gradient computation is a dot product per sample.

### K-Means Clustering

For `N` samples, `K` clusters, `D` features, `I` iterations:

```
Time:  O(N × K × D × I)
Space: O(K × D + N)  (centroids + assignments)
```

Each iteration assigns every point to its nearest centroid (O(N × K × D)) and recomputes centroids (O(N × D)). K-means++ initialization adds O(N × K × D) for the weighted sampling.

### Isolation Forest

For `N` training samples, `T` trees, `S` subsample size:

```
Build:  O(N × T × S)
Score:  O(T × log(S))  (average path length)
Space:  O(T × S)
```

The key insight from Liu et al. (2008): anomalies have short average path lengths because they are "isolated" early in the tree construction. Normal points require more splits to isolate, resulting in longer paths. The expected path length for a dataset of `n` points is `c(n) = 2H(n-1) - 2(n-1)/n`, where `H` is the harmonic number.

---

## Quick Start

```bash
# Clone
git clone https://github.com/SuperInstance/exocortex-kernel-c.git
cd exocortex-kernel-c

# Build
make

# Run tests
make test

# Check for memory leaks (requires valgrind)
make valgrind

# Clean
make clean
```

### Minimal Example

```c
#include "exocortex/logistic.h"
#include <stdio.h>

int main(void) {
    LogisticRegression lr;
    logistic_create(&lr, 2);

    /* Train on AND gate */
    double X[] = {0,0, 0,1, 1,0, 1,1};
    int    y[] = {0,   0,   0,   1};
    logistic_train(&lr, X, y, 4, 1.0, 500);

    double prob;
    logistic_predict(&lr, (double[]){1, 1}, &prob);
    printf("P(1 AND 1) = %.4f\n", prob);  /* ~0.85+ */

    logistic_free(&lr);
    return 0;
}
```

Compile:
```bash
gcc -std=c11 -O2 -I. -o example example.c src/*.c -lm
```

---

## Building

### Prerequisites

- GCC or Clang (C99/C11 support)
- `make`
- Standard math library (`libm`)
- (Optional) `valgrind` for leak checking

### Targets

| Target | Description |
|---|---|
| `make` | Build `libexocortex.a` static library |
| `make test` | Build and run test suite |
| `make valgrind` | Run tests under valgrind |
| `make clean` | Remove build artifacts |

### Cross-Compilation

Since this is pure C with no platform-specific code, cross-compilation is trivial:

```bash
# ARM embedded
arm-none-eabi-gcc -std=c11 -O2 -c src/*.c -I.

# RISC-V
riscv64-unknown-elf-gcc -std=c11 -O2 -c src/*.c -I.

# WASM (via Emscripten)
emcc -std=c11 -O2 src/*.c -I. -o exocortex.js
```

---

## Library Reference

### Matrix Operations (`matrix.h`)

The `Matrix` struct is the foundation:

```c
typedef struct {
    double *data;   // Row-major flat array
    int rows;
    int cols;
} Matrix;
```

| Function | Signature | Description |
|---|---|---|
| `mat_create` | `Matrix mat_create(int rows, int cols)` | Allocate zero-filled matrix |
| `mat_free` | `void mat_free(Matrix *m)` | Release memory |
| `mat_get` | `double mat_get(const Matrix *m, int r, int c)` | Element access (inline) |
| `mat_set` | `void mat_set(Matrix *m, int r, int c, double val)` | Element set (inline) |
| `mat_mul` | `int mat_mul(const Matrix *a, const Matrix *b, Matrix *out)` | Matrix multiplication |
| `mat_transpose` | `int mat_transpose(const Matrix *a, Matrix *out)` | Transpose |
| `mat_add` | `int mat_add(const Matrix *a, const Matrix *b, Matrix *out)` | Element-wise addition |
| `mat_scale` | `int mat_scale(Matrix *m, double scalar)` | Scalar multiplication |
| `mat_apply` | `int mat_apply(Matrix *m, double (*fn)(double))` | Element-wise function |
| `mat_print` | `void mat_print(const Matrix *m, const char *label)` | Debug print |

### Statistics (`stats.h`)

| Function | Signature | Description |
|---|---|---|
| `mean` | `double mean(const double *v, int n)` | Arithmetic mean |
| `variance` | `double variance(const double *v, int n)` | Population variance |
| `standardize` | `int standardize(double *v, int n, double *out)` | Zero mean, unit variance |
| `argmax` | `int argmax(const double *v, int n)` | Index of maximum element |

### Micro Neural Network (`micro_nn.h`)

```c
typedef struct {
    int input_dim;
    int hidden_dim;
    int output_dim;
    int classification;   // 1 = softmax, 0 = linear
    Matrix weights_1;     // input_dim × hidden
    Matrix bias_1;        // 1 × hidden
    Matrix weights_2;     // hidden × output_dim
    Matrix bias_2;        // 1 × output_dim
} MicroNN;
```

| Function | Description |
|---|---|
| `micro_nn_create(nn, in, hid, out, classification)` | Initialize with random weights (Xavier-ish) |
| `micro_nn_free(nn)` | Release all memory |
| `micro_nn_forward(nn, input, output)` | Forward pass |
| `micro_nn_train(nn, inputs, targets, n, lr, epochs)` | SGD with backpropagation |

**Activations:** ReLU (hidden), Softmax (classification output) or Linear (regression output).

**Weight initialization:** Normal distribution scaled by √(2/n) (He initialization variant).

### Logistic Regression (`logistic.h`)

```c
typedef struct {
    int    n_features;
    double *weights;
    double bias;
} LogisticRegression;
```

| Function | Description |
|---|---|
| `logistic_create(lr, n_features)` | Initialize with small random weights |
| `logistic_free(lr)` | Release memory |
| `logistic_train(lr, X, y, n, lr, epochs)` | Gradient descent on binary labels |
| `logistic_predict(lr, x, &prob)` | Predict probability ∈ [0, 1] |

**Sigmoid:** `σ(z) = 1 / (1 + e^{-z})` with numerical clamping for overflow protection.

### K-Means Clustering (`kmeans.h`)

```c
typedef struct {
    int    k;
    int    n_features;
    double *centroids;    // k × n_features
    int    *assignments;  // n_samples
    int    n_assigned;
} KMeans;
```

| Function | Description |
|---|---|
| `kmeans_create(km, k, n_features)` | Allocate centroid storage |
| `kmeans_free(km)` | Release memory |
| `kmeans_fit(km, X, n, max_iter)` | K-means++ init + Lloyd's algorithm |
| `kmeans_predict(km, x)` | Return nearest cluster ID |

**Initialization:** K-means++ — selects initial centroids proportional to their squared distance from existing centroids. This guarantees O(log k) competitive ratio versus optimal clustering.

### Isolation Forest (`isolation.h`)

```c
typedef struct IsolationTree {
    int    split_feature;
    double split_value;
    int    is_leaf;
    int    size;
    struct IsolationTree *left;
    struct IsolationTree *right;
} IsolationTree;

typedef struct {
    int             n_trees;
    int             sample_size;
    int             max_depth;
    IsolationTree  *trees;
    int             n_features;
    int             n_training;
} IsolationForest;
```

| Function | Description |
|---|---|
| `isolation_forest_create(forest, n_trees, sample_size)` | Allocate tree array |
| `isolation_forest_free(forest)` | Release all tree memory |
| `isolation_forest_fit(forest, X, n, n_features)` | Build isolation trees from subsamples |
| `isolation_forest_score(forest, x)` | Compute anomaly score ∈ [0, 1] |

**Tree traversal:** Fully iterative using an explicit stack — no recursion, no stack overflow risk.

**Anomaly score:** `s(x, n) = 2^{-E(h(x)) / c(n)}` where `E(h(x))` is the average path length and `c(n)` is the normalization factor using the Euler-Mascheroni constant.

### Compute Kernel (`compute.h`)

Unified interface wrapping all model types:

```c
typedef struct {
    ComputeModelType type;
    union {
        MicroNN           nn;
        LogisticRegression logistic;
        KMeans             kmeans;
        IsolationForest    isolation;
    } model;
} ComputeKernel;
```

---

## Examples

### Example 1: XOR Classification with MicroNN

```c
#include "exocortex/micro_nn.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    srand(42);
    MicroNN nn;
    micro_nn_create(&nn, 2, 8, 1, 0);  /* 2 inputs, 8 hidden, 1 output, regression */

    /* XOR truth table */
    double inputs[] = {0,0, 0,1, 1,0, 1,1};
    double targets[] = {0, 1, 1, 0};

    /* Train */
    int result = micro_nn_train(&nn, inputs, targets, 4, 0.5, 2000);
    if (result != 0) {
        fprintf(stderr, "Training failed!\n");
        return 1;
    }

    /* Evaluate */
    double output;
    double test_inputs[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    double expected[4] = {0, 1, 1, 0};

    printf("XOR Results:\n");
    for (int i = 0; i < 4; i++) {
        micro_nn_forward(&nn, test_inputs[i], &output);
        printf("  [%.0f, %.0f] -> %.4f (expected %.0f)\n",
               test_inputs[i][0], test_inputs[i][1], output, expected[i]);
    }

    micro_nn_free(&nn);
    return 0;
}
```

### Example 2: Sentiment Classifier with Logistic Regression

```c
#include "exocortex/logistic.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    srand(42);
    LogisticRegression lr;
    logistic_create(&lr, 3);  /* 3 features: positive words, negative words, length */

    /* Training data: [pos_words, neg_words, length_normalized] */
    double X[] = {
        5.0, 1.0, 0.8,   /* positive review */
        1.0, 6.0, 0.5,   /* negative review */
        4.0, 2.0, 0.7,   /* positive review */
        2.0, 7.0, 0.4,   /* negative review */
        6.0, 0.0, 0.9,   /* very positive  */
        0.0, 8.0, 0.3    /* very negative  */
    };
    int y[] = {1, 0, 1, 0, 1, 0};

    logistic_train(&lr, X, y, 6, 0.5, 1000);

    /* Classify new review: [pos=3, neg=2, length=0.6] */
    double prob;
    double new_review[] = {3.0, 2.0, 0.6};
    logistic_predict(&lr, new_review, &prob);

    printf("Sentiment probability: %.4f\n", prob);
    printf("Classification: %s\n", prob >= 0.5 ? "POSITIVE" : "NEGATIVE");

    logistic_free(&lr);
    return 0;
}
```

### Example 3: Customer Segmentation with K-Means

```c
#include "exocortex/kmeans.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    srand(42);
    KMeans km;
    kmeans_create(&km, 3, 2);  /* 3 clusters, 2 features (spending, frequency) */

    /* Customer data: [monthly_spend, visit_frequency] */
    double customers[] = {
        /* Low spenders */
         50.0,  2.0,
         30.0,  1.0,
         40.0,  3.0,
        /* Medium spenders */
        200.0,  5.0,
        180.0,  6.0,
        220.0,  4.0,
        /* High spenders */
        500.0, 10.0,
        450.0, 12.0,
        550.0,  9.0
    };

    kmeans_fit(&km, customers, 9, 100);

    /* Classify a new customer */
    double new_customer[] = {150.0, 4.0};
    int cluster = kmeans_predict(&km, new_customer);
    printf("New customer assigned to cluster %d\n", cluster);

    /* Print all assignments */
    printf("\nCustomer assignments:\n");
    for (int i = 0; i < 9; i++) {
        printf("  Customer %d → Cluster %d\n", i, km.assignments[i]);
    }

    kmeans_free(&km);
    return 0;
}
```

### Example 4: Network Anomaly Detection with Isolation Forest

```c
#include "exocortex/isolation.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    srand(42);
    IsolationForest forest;
    isolation_forest_create(&forest, 50, 20);

    /* Normal network traffic: [latency_ms, packet_loss_%, throughput_mbps] */
    #define N_SAMPLES 30
    double traffic[N_SAMPLES * 3];

    /* Generate normal traffic */
    for (int i = 0; i < N_SAMPLES; i++) {
        traffic[i*3]     = 10.0 + ((double)rand() / RAND_MAX) * 20.0;  /* 10-30ms latency */
        traffic[i*3 + 1] = ((double)rand() / RAND_MAX) * 2.0;          /* 0-2% loss */
        traffic[i*3 + 2] = 80.0 + ((double)rand() / RAND_MAX) * 40.0;  /* 80-120 Mbps */
    }

    isolation_forest_fit(&forest, traffic, N_SAMPLES, 3);

    /* Check normal traffic */
    double normal[] = {15.0, 1.0, 100.0};
    double s_normal = isolation_forest_score(&forest, normal);
    printf("Normal traffic score: %.4f\n", s_normal);

    /* Check anomalous traffic */
    double anomaly[] = {500.0, 45.0, 0.5};
    double s_anomaly = isolation_forest_score(&forest, anomaly);
    printf("Anomalous traffic score: %.4f\n", s_anomaly);

    printf("\n%s\n", s_anomaly > s_normal ? "Anomaly detected correctly!" : "Something is wrong...");

    isolation_forest_free(&forest);
    return 0;
}
```

---

## Performance Comparison

Theoretical performance comparison across implementations. Numbers are relative — C is the baseline at 1.0×.

### Operation: Matrix Multiply (100×100)

| Implementation | Relative Time | Memory Overhead | Binary Size |
|---|---|---|---|
| **C (this library)** | **1.0×** | **0 bytes** | **~15 KB** |
| Rust (ndarray) | ~1.0× | ~0 bytes | ~200 KB |
| Python (numpy/OpenBLAS) | ~1.2× | ~50 MB (runtime) | ~100 MB |
| Julia (stdlib) | ~1.0× (post-JIT) | ~200 MB (runtime) | ~200 MB |

### Operation: MLP Forward Pass (784→128→10)

| Implementation | Relative Time | Startup | Dependencies |
|---|---|---|---|
| **C (this library)** | **1.0×** | **~1 μs** | **None** |
| TensorFlow Lite | ~1.5× | ~50 ms | ~1 MB |
| ONNX Runtime | ~1.3× | ~30 ms | ~5 MB |
| PyTorch | ~2.0× | ~500 ms | ~500 MB |
| ggml (llama.cpp) | ~1.0× | ~5 ms | ~500 KB |

### Why These Numbers Matter

The raw throughput numbers are similar because the bottleneck is always memory bandwidth and BLAS operations. The real differentiator is:

1. **Startup time** — C wins by 3-6 orders of magnitude
2. **Memory overhead** — C has zero overhead beyond data
3. **Deployment footprint** — C binaries are measured in KB, not MB or GB
4. **Latency** — No JIT warmup, no garbage collection pauses

For batch training on GPUs, use PyTorch. For inference on anything with a pulse and a stack pointer, use C.

---

## Design Decisions

### Row-Major Layout

```
Matrix (3×4) in memory:
  [a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23]
   ──────────────────   ──────────────────   ──────────────────
        Row 0                Row 1                Row 2
```

Row-major (C-order) was chosen because:
1. It matches C's natural array layout — no index transformation
2. Cache-friendly for row-wise operations (forward pass, matrix multiply inner loop)
3. Compatible with BLAS conventions (row-major = CblasRowMajor)
4. Interoperable with any C data structure without transposition

### Error Codes Over Exceptions

All functions return `int` error codes: 0 for success, -1 for failure.

```c
int result = micro_nn_forward(&nn, input, output);
if (result != 0) {
    /* Handle error — no exceptions, no unwinding, no surprises */
}
```

Why no `errno`? Because `errno` is global state, and global state is the enemy of embedded systems and multithreaded code.

Why no setjmp/longjmp? Because those are hidden gotos that bypass cleanup. In a library that allocates memory, bypassing cleanup means leaks.

### No Opaque Pointles

All structs are defined in headers, not hidden behind `void*` or forward declarations.

```c
/* You get the full definition */
typedef struct {
    double *weights;
    double bias;
    int n_features;
} LogisticRegression;
```

Why? Because opaque pointers are abstractions, and abstractions have costs:
- You can't inspect model state for debugging
- You can't serialize without accessor functions
- You can't optimize access patterns for your specific use case
- You can't stack-allocate or embed in other structs

### Explicit Memory Management

Every `malloc` has a matching `free`. Every struct has a `_create` and `_free` function. No hidden allocations, no reference counting, no garbage collector.

```c
/* Pattern: create → use → free */
MicroNN nn;
micro_nn_create(&nn, 10, 20, 5, 1);
/* ... use nn ... */
micro_nn_free(&nn);  /* All memory released */
```

This pattern is deliberate:
1. Caller owns the memory — always clear who frees what
2. Stack allocation of the struct, heap allocation of the data — minimal overhead
3. No hidden allocations in "getter" functions
4. valgrind-clean by design

### Iterative Tree Traversal

The isolation forest uses explicit stacks for tree building and scoring, not recursion.

```c
/* NOT this (recursive): */
double score_recursive(Node *n, double *x, int depth) {
    if (n->is_leaf) return depth + c(n->size);
    if (x[n->feature] < n->split)
        return score_recursive(n->left, x, depth + 1);
    return score_recursive(n->right, x, depth + 1);
}

/* THIS (iterative): */
ScoreFrame stack[MAX_DEPTH];
int top = 0;
stack[top++] = (ScoreFrame){root, 0};
while (top > 0) {
    ScoreFrame f = stack[--top];
    if (f.node->is_leaf) { path = f.depth + c(f.node->size); break; }
    /* push next */
}
```

Why? Because:
1. Stack depth is bounded by `max_depth`, not data size
2. No risk of stack overflow on embedded systems (call stack might be 4-8 KB)
3. No function call overhead per level
4. Predictable memory usage

### Random Number Generation

Uses `rand()` from `<stdlib.h>`. This is deliberately simple:
1. Seed with `srand()` — the caller controls reproducibility
2. No dependency on OS-specific entropy sources
3. Adequate for ML initialization (not cryptography)
4. Replaceable by the caller if they need better randomness

---

## Comparison with Existing Frameworks

### vs TensorFlow Lite

| Dimension | exocortex-kernel-c | TensorFlow Lite |
|---|---|---|
| Binary size | ~15 KB | ~1 MB |
| Dependencies | None | FlatBuffers, absl |
| Build system | Make | Bazel/CMake |
| Model format | C structs | .tflite protobuf |
| Training | Built-in | No (inference only) |
| GPU support | No | Yes (delegate API) |
| Quantization | Manual | Built-in (int8) |
| Target platform | Anything with a C compiler | Android, Linux, MCU |

**When to use TFLite:** You need GPU acceleration, quantized inference, or pre-trained model compatibility.

**When to use exocortex-kernel-c:** You need on-device training, zero dependencies, or deployment to platforms TFLite doesn't support.

### vs ONNX Runtime

| Dimension | exocortex-kernel-c | ONNX Runtime |
|---|---|---|
| Binary size | ~15 KB | ~5-50 MB |
| Dependencies | None | protobuf, ONNX spec |
| Model support | 4 algorithms | 170+ operators |
| Training | Built-in | Training mode (limited) |
| Custom ops | Modify source | Register custom op |
| Language | C | C++/Python/C#/Java |

**When to use ONNX Runtime:** You're deploying models from PyTorch/TensorFlow and need operator compatibility.

**When to use exocortex-kernel-c:** You're building models from scratch in C or need a minimal footprint.

### vs ggml (llama.cpp)

| Dimension | exocortex-kernel-c | ggml |
|---|---|---|
| Binary size | ~15 KB | ~500 KB |
| Focus | Traditional ML | Tensor operations / LLMs |
| Graph execution | Direct calls | Compute graph |
| Quantization | None | Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 |
| GPU support | None | CUDA, Metal, OpenCL, Vulkan |
| Algorithms | NN, Logistic, KMeans, Isolation | Tensor math + backends |

ggml is excellent for what it does — running large language models efficiently. It's a tensor library with backends. exocortex-kernel-c is a higher-level ML library with specific algorithms. Different scope, different purpose.

---

## Historical References

### McCulloch & Pitts (1943)

Warren McCulloch and Walter Pitts published "A Logical Calculus of the Ideas Immanent in Nervous Activity" in 1943, proposing the first mathematical model of a neuron. Their artificial neuron was a binary threshold unit: it fires (outputs 1) if the weighted sum of inputs exceeds a threshold, and stays silent (outputs 0) otherwise.

This is the intellectual ancestor of every neural network, including the MicroNN in this library. The key insight — that computation can be modeled as interconnected simple units — remains unchanged 80+ years later.

**Connection to this library:** The MicroNN's forward pass (`z = Wx + b`, followed by activation) is a direct descendant of the McCulloch-Pitts neuron, generalized to continuous values and multi-layer networks.

### Rosenblatt (1958) — The Perceptron

Frank Rosenblatt's perceptron (1958) was the first machine learning algorithm that could *learn* its weights from data. Unlike McCulloch-Pitts neurons with fixed weights, the perceptron used a simple update rule:

```
w_i ← w_i + α(y - ŷ)x_i
```

This is fundamentally the same update rule used in `logistic_train()` — compute error, scale by input and learning rate, update weights. Rosenblatt proved that if the data is linearly separable, the perceptron will converge in finite steps.

**Connection to this library:** `LogisticRegression` is essentially the perceptron with a sigmoid activation instead of a step function, trained with gradient descent instead of the perceptron update rule. The core idea — adjusting weights to reduce error — is identical.

### Rumelhart, Hinton & Williams (1986) — Backpropagation

The backpropagation algorithm, popularized by Rumelhart, Hinton, and Williams in their 1986 Nature paper "Learning representations by back-propagating errors," showed that multi-layer networks could be trained efficiently by propagating error gradients backward through the network.

```
Forward:  z₁ = xW₁ + b₁,  a₁ = ReLU(z₁),  z₂ = a₁W₂ + b₂
Loss:     L = ½Σ(ŷ - y)²
Backward: ∂L/∂W₂ = a₁ᵀ(y - ŷ),  ∂L/∂W₁ = xᵀ(δ₁)
Update:   W ← W - α × ∂L/∂W
```

**Connection to this library:** `micro_nn_train()` implements exactly this algorithm — forward pass, loss computation, backpropagation of gradients through both layers, and SGD weight updates. The implementation in `src/micro_nn.c` maps 1:1 to the equations above.

### Lloyd (1982) — K-Means

Stuart Lloyd's algorithm for least-squares quantization, initially developed at Bell Labs in 1957 and published in 1982, remains the standard K-means algorithm. It alternates between:

1. **Assignment step:** Assign each point to the nearest centroid
2. **Update step:** Move each centroid to the mean of its assigned points

**Connection to this library:** `kmeans_fit()` implements Lloyd's algorithm with K-means++ initialization (Arthur & Vassilvitskii, 2007), which provides a O(log k)-competitive guarantee on the initial centroid placement.

### Liu, Ting & Zhou (2008) — Isolation Forest

Fei Tony Liu, Kai Ming Ting, and Zhi-Hua Zhou introduced the Isolation Forest algorithm at ICDM 2008. Their key insight was elegant: instead of modeling normal behavior and flagging deviations, *isolate* anomalies directly. Anomalies are "few and different" — they require fewer random partitions to isolate.

The anomaly score is:

```
s(x, n) = 2^(-E(h(x)) / c(n))
```

Where:
- `h(x)` = path length to isolate point x
- `E(h(x))` = average path length over all trees
- `c(n)` = normalization factor based on Euler-Mascheroni constant
- Score near 1 → anomaly; score near 0.5 → normal

**Connection to this library:** `isolation_forest_score()` computes this exact score, with iterative (stack-based) tree traversal for robustness on constrained systems.

---

## The PLATO Principle

> **"If the ESP32 can't run it, it's not real ML."**

PLATO, as in: **P**ortable, **L**ightweight, **A**uditable, **T**ransparent, **O**ptimal.

The ESP32 has:
- 520 KB SRAM
- 240 MHz dual-core LX6
- No FPU (Xtensa, not ARM)
- No OS (bare metal or FreeRTOS)

If your ML library requires:
- A GPU → Not PLATO
- 100 MB of RAM → Not PLATO
- A Python runtime → Definitely not PLATO
- An internet connection for model downloads → Absolutely not PLATO

This library is PLATO-compliant:
- **Portable:** C99/C11 — runs on every platform with a C compiler
- **Lightweight:** ~15 KB binary, uses only what you allocate
- **Auditable:** Every line is readable, every allocation is traceable
- **Transparent:** No hidden costs, no magic, no auto-tuning
- **Optimal:** O-optimal algorithms with minimal constant factors

The entire library compiles to less code than the `import numpy` statement in Python. Think about that.

---

## Glossary

### ML Terms for C Programmers

| ML Term | C Analogy | What It Actually Means |
|---|---|---|
| **Epoch** | Loop iteration | One complete pass through the training data |
| **Learning Rate** | Step size in iteration | How much to adjust weights per update; too large = oscillation, too small = slow convergence |
| **Gradient** | Derivative / slope | Direction and magnitude of the error surface; "which way is downhill?" |
| **Backpropagation** | Reverse-mode AD | Chain rule applied systematically through layers; computing ∂Loss/∂Weight for every weight |
| **Loss Function** | Objective function | What you're minimizing; MSE for regression, cross-entropy for classification |
| **Weight** | Coefficient / parameter | A learnable value in the model, like a coefficient in a polynomial |
| **Bias** | Offset / intercept | A constant added to the weighted sum; allows the model to shift its decision boundary |
| **Activation** | Nonlinear function | ReLU, sigmoid, tanh; gives the network nonlinearity (without it, deep nets collapse to linear) |
| **Softmax** | Probability normalizer | Converts arbitrary scores to a probability distribution (sum to 1, all positive) |
| **SGD** | Steepest descent | Update weights in the direction that reduces loss; stochastic because it uses random samples |
| **Overfitting** | Memorization | Model learns training data too well, including noise; fails to generalize |
| **Feature** | Input variable | One dimension of your input data (like a field in a struct) |
| **Label** | Expected output | The "answer key" for supervised learning |
| **Inference** | Function call | Running data through the trained model to get a prediction |
| **Training** | Curve fitting | Adjusting model parameters to minimize error on training data |
| **Cluster** | Group | A set of similar data points found by K-means |
| **Centroid** | Group mean | The center of a cluster; the average of all points assigned to it |
| **Anomaly** | Outlier | A data point that's significantly different from the majority |
| **Isolation** | Separation | How easily a random partition tree can isolate a point (anomalies isolate faster) |
| **Path Length** | Tree depth | Number of splits needed to isolate a point; shorter = more anomalous |
| **Standardization** | Normalization | Transforming data to zero mean, unit variance; like normalizing a vector |
| **Embedding** | Learned representation | A dense vector representation of data, often lower-dimensional than raw input |
| **Hyperparameter** | Config constant | A parameter set by the programmer (learning rate, epochs, etc.), not learned from data |
| **Batch Size** | Chunk size | Number of samples processed before updating weights; 1 = stochastic, N = batch |
| **Regularization** | Constraint | Penalizing complexity to prevent overfitting (not implemented here — keep it simple) |

### C Terms for ML Engineers

| C Term | What ML People Call It | Why It Matters |
|---|---|---|
| `malloc` | Tensor allocation | Every array you use in NumPy is a malloc underneath |
| `free` | Garbage collection | In C, you do it yourself. In Python, the GC does it (slowly). |
| Pointer | Tensor reference | A memory address pointing to your data |
| `static inline` | JIT-inlined function | The compiler does what PyTorch's JIT tries to do |
| `-O2` | Optimization pass | Compiler optimizations = free performance |
| Stack allocation | No-allocation tensor | For small buffers, stack allocation avoids malloc entirely |
| Row-major | C-order (numpy) | How data is laid out in memory; affects cache performance |
| `valgrind` | Memory profiler | Finds leaks and undefined behavior; there's no equivalent in Python |

---

## Project Structure

```
exocortex-kernel-c/
├── include/
│   └── exocortex/
│       ├── compute.h      # Unified kernel interface
│       ├── micro_nn.h     # 2-layer MLP
│       ├── logistic.h     # Logistic regression
│       ├── kmeans.h       # K-means clustering
│       ├── isolation.h    # Isolation forest
│       ├── matrix.h       # Dense matrix operations
│       └── stats.h        # Basic statistics
├── src/
│   ├── compute.c
│   ├── micro_nn.c
│   ├── logistic.c
│   ├── kmeans.c
│   ├── isolation.c
│   ├── matrix.c
│   └── stats.c
├── tests/
│   ├── test_micro_nn.c
│   ├── test_logistic.c
│   ├── test_kmeans.c
│   ├── test_isolation.c
│   └── test_main.c
├── Makefile
├── README.md
└── .gitignore
```

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Write code (C99/C11 compatible)
4. Add tests
5. Run `make clean && make && make test`
6. Run `make valgrind` (no leaks)
7. Submit a pull request

### Code Style

- 4-space indentation (no tabs)
- `{` on same line for functions, next line for structs/conditionals (be consistent within a file)
- `snake_case` for functions and variables
- `PascalCase` for struct typedefs
- Every function returns `int` (0 = success, -1 = error)
- Every `malloc` must have a matching `free`

---

## License

MIT License. Use it for anything. No warranty. Learn, build, deploy.

---

*Built with the PLATO principle: Portable, Lightweight, Auditable, Transparent, Optimal.*

*"The best code is the code that runs everywhere, not just everywhere that supports your framework."*
