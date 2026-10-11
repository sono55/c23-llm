#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ============================================================================
// VERSION SPECIFICATION (C23 constexpr)
// ============================================================================
constexpr int VERSION_MAJOR = 1;
constexpr int VERSION_MINOR = 0;

// ============================================================================
// 1. TOPOLOGY SPECIFICATION (C23 constexpr)
// ============================================================================
// constexpr evaluates strictly at compile-time, allowing fixed-size buffers
// without resorting to heap allocation or Variable Length Arrays (VLAs).
constexpr size_t N_IN  = 2; // Input coordinates: (x, y)
constexpr size_t N_H1  = 8; // Hidden Layer 1: 8 nodes
constexpr size_t N_H2  = 8; // Hidden Layer 2: 8 nodes
constexpr size_t N_OUT = 1; // Output Layer: 1 node (binary probability)

// Matrix dimensions (weights connect every input to every output node):
constexpr size_t W1_SIZE = N_H1 * N_IN;  // 8 * 2 = 16 weights
constexpr size_t W2_SIZE = N_H2 * N_H1;  // 8 * 8 = 64 weights
constexpr size_t W3_SIZE = N_OUT * N_H2; // 1 * 8 = 8 weights

// ----------------------------------------------------------------------------
// MEMORY SEPARATION ARCHITECTURE:
// - NeuralNetwork: Read-only model parameters (weights & biases).
//   Can be shared safely across all threads during inference.
// - NetworkActivations: Transient scratchpad memory for a single query.
//   Allocated on the local stack; avoids race conditions in multi-threading.
// ----------------------------------------------------------------------------
typedef struct {
    // Weights stored in flat, contiguous row-major order: node_j_weights[i]
    float w1[W1_SIZE];
    float b1[N_H1];

    float w2[W2_SIZE];
    float b2[N_H2];

    float w3[W3_SIZE];
    float b3[N_OUT];
} NeuralNetwork;

typedef struct {
    float h1[N_H1];
    float h2[N_H2];
    float out[N_OUT];
} NetworkActivations;

// ============================================================================
// 2. ACTIVATION FUNCTIONS & UTILITIES
// ============================================================================

// WEIGHT INITIALIZATION STRATEGY:
// Small random floats in [-0.5, +0.5]:
// 1. Must NOT be all zero: Breaks node symmetry so all 8 nodes don't clone.
// 2. Must NOT be large: Prevents Sigmoid from saturating where slope is zero.
static inline float random_weight(void) 
{
    return ((float)rand() / (float)RAND_MAX) - 0.5f;
}

// HIDDEN LAYER ACTIVATION (ReLU):
// - Math: f(z) = max(0, z). Acts like a hinge that bends linear space.
// - Why inside? Blazingly fast (1 clock cycle) and derivative is 1.0,
//   which completely prevents the "vanishing gradient" problem in deep layers.
static inline float relu(float z) 
{
    return (z > 0.0f) ? z : 0.0f;
}

// OUTPUT LAYER ACTIVATION (Sigmoid):
// - Math: 1 / (1 + e^-z).
// - Why at the exit? Squashes arbitrary sums (-inf to +inf) into a clean
//   probability in [0.0, 1.0], where 0.5 is the exact 50/50 decision boundary.
static inline float sigmoid(float z) 
{
    return 1.0f / (1.0f + expf(-z));
}

// ============================================================================
// 3. CORE COMPUTATION: DENSE LAYER
// ============================================================================
// Computes: out = Activation(weights * in + biases)
//
// 'restrict' keyword (C23/C99): Promises to the compiler that in, out, weights,
// and biases never alias (overlap) in memory. This unlocks AVX2 SIMD vectorization.
//
// Memory stride: Row-major order.
// Node j's weights start at index: (j * in_dim).
static void dense_forward(
    const float *restrict in,
    float *restrict out,
    const float *restrict weights,
    const float *restrict biases,
    size_t in_dim,
    size_t out_dim,
    bool use_sigmoid
) {
    for (size_t j = 0; j < out_dim; j++) {
        // Start with the node's threshold offset (bias):
        float sum = biases[j];

        // Compute dot-product with incoming inputs:
        const size_t row_offset = j * in_dim;
        for (size_t i = 0; i < in_dim; i++) {
            sum += in[i] * weights[row_offset + i];
        }

        // Apply layer-appropriate activation:
        out[j] = use_sigmoid ? sigmoid(sum) : relu(sum);
    }
}

// Complete Forward Pass from (x, y) coordinates to final probability
float forward_pass(const NeuralNetwork *restrict net, 
                   NetworkActivations *restrict acts, 
                   const float input[N_IN]) 
{
    // Layer 1: Input (2) -> Hidden 1 (8) with ReLU
    dense_forward(input, acts->h1, net->w1, net->b1, N_IN, N_H1, false);

    // Layer 2: Hidden 1 (8) -> Hidden 2 (8) with ReLU
    dense_forward(acts->h1, acts->h2, net->w2, net->b2, N_H1, N_H2, false);

    // Layer 3: Hidden 2 (8) -> Output (1) with Sigmoid
    dense_forward(acts->h2, acts->out, net->w3, net->b3, N_H2, N_OUT, true);

    return acts->out[0];
}

// ============================================================================
// 4. INITIALIZATION & INSPECTION
// ============================================================================
void init_network(NeuralNetwork *net) 
{
    // Seed PRNG with a fixed constant (42) for deterministic, reproducible runs
    srand(42);

    for (size_t i = 0; i < W1_SIZE; i++) net->w1[i] = random_weight();
    for (size_t i = 0; i < N_H1; i++)    net->b1[i] = 0.0f;

    for (size_t i = 0; i < W2_SIZE; i++) net->w2[i] = random_weight();
    for (size_t i = 0; i < N_H2; i++)    net->b2[i] = 0.0f;

    for (size_t i = 0; i < W3_SIZE; i++) net->w3[i] = random_weight();
    for (size_t i = 0; i < N_OUT; i++)   net->b3[i] = 0.0f;
}

// Nicely formats and prints all 4 layers, weights, and biases to stdout
void print_network_parameters(const NeuralNetwork *net) 
{
    printf("\n================================================================\n");
    printf("        CURRENT NETWORK PARAMETERS (Weights & Biases)            \n");
    printf("================================================================\n");

    // --- Layer 1: Input (2) -> Hidden 1 (8) ---
    printf("\n[Layer 1: Input (2) -> Hidden 1 (8)]\n");
    for (size_t j = 0; j < N_H1; j++) {
        printf("  Node %zu | Bias: %+6.3f | Weights: [ ", j, net->b1[j]);
        for (size_t i = 0; i < N_IN; i++) {
            printf("%+6.3f ", net->w1[j * N_IN + i]);
        }
        printf("]\n");
    }

    // --- Layer 2: Hidden 1 (8) -> Hidden 2 (8) ---
    printf("\n[Layer 2: Hidden 1 (8) -> Hidden 2 (8)]\n");
    for (size_t j = 0; j < N_H2; j++) {
        printf("  Node %zu | Bias: %+6.3f | Weights: [ ", j, net->b2[j]);
        for (size_t i = 0; i < N_H1; i++) {
            printf("%+6.3f ", net->w2[j * N_H1 + i]);
        }
        printf("]\n");
    }

    // --- Layer 3: Hidden 2 (8) -> Output (1) ---
    printf("\n[Layer 3: Hidden 2 (8) -> Output (1)]\n");
    for (size_t j = 0; j < N_OUT; j++) {
        printf("  Node %zu | Bias: %+6.3f | Weights: [ ", j, net->b3[j]);
        for (size_t i = 0; i < N_H2; i++) {
            printf("%+6.3f ", net->w3[j * N_H2 + i]);
        }
        printf("]\n");
    }
    printf("================================================================\n\n");
}

// ============================================================================
// 5. TERMINAL VISUALIZER (The ASCII Grid)
// ============================================================================
// Why 861 evaluations?
// A neural network is a mathematical function f(x, y), not an image canvas.
// Plotting a 2D map across 21x41 pixels requires querying f(x, y) at 861 points.
void render_decision_boundary(const NeuralNetwork *net) 
{
    NetworkActivations acts = {}; // C23 empty zero-initialization

    constexpr int ROWS = 21;
    constexpr int COLS = 41;

    printf("--- DECISION BOUNDARY (Field View - V%d.%d) ---\n", VERSION_MAJOR, VERSION_MINOR);
    for (int r = 0; r < ROWS; r++) {
        // Map row [0..20] to Cartesian y: +1.0 down to -1.0
        float y = 1.0f - (2.0f * (float)r / (float)(ROWS - 1));

        for (int c = 0; c < COLS; c++) {
            // Map column [0..40] to Cartesian x: -1.0 to +1.0
            float x = -1.0f + (2.0f * (float)c / (float)(COLS - 1));

            float input[N_IN] = { x, y };
            float prob = forward_pass(net, &acts, input);

            // Print '#' if network predicts inside circle (>= 0.5), else '.'
            if (prob >= 0.5f) {
                putchar('#');
            } else {
                putchar('.');
            }
        }
        putchar('\n');
    }
    printf("Legend: '#' = INSIDE (>= 0.5), '.' = OUTSIDE (< 0.5)\n\n");
}

int main(void) 
{
    NeuralNetwork net = {}; // C23 empty zero-initialization
    init_network(&net);

    // Print summary stats and initial parameters
    printf("Neural Network V%d.%d Initialized:\n", VERSION_MAJOR, VERSION_MINOR);
    printf("Total Parameters: %zu floats (%zu weights, %zu biases)\n", 
           W1_SIZE + W2_SIZE + W3_SIZE + N_H1 + N_H2 + N_OUT,
           W1_SIZE + W2_SIZE + W3_SIZE,
           N_H1 + N_H2 + N_OUT);

    // Display the initial weights and biases:
    print_network_parameters(&net);

    // Test a single query at the center origin (0.0, 0.0)
    NetworkActivations test_acts = {};
    float center_point[N_IN] = { 0.0f, 0.0f };
    float center_prob = forward_pass(&net, &test_acts, center_point);
    printf("Prediction at center (0.0, 0.0): %.4f (Probability)\n\n", center_prob);

    // Render the untrained field map:
    render_decision_boundary(&net);

    return 0;
}