#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ============================================================================
// 1. TOPOLOGY SPECIFICATION (C23 constexpr named constants)
// ============================================================================
constexpr size_t N_IN  = 2; // (x, y) coordinates
constexpr size_t N_H1  = 8; // Hidden Layer 1
constexpr size_t N_H2  = 8; // Hidden Layer 2
constexpr size_t N_OUT = 1; // Output probability

// Exact array dimensions evaluated at compile time:
constexpr size_t W1_SIZE = N_H1 * N_IN;  // 8 * 2 = 16
constexpr size_t W2_SIZE = N_H2 * N_H1;  // 8 * 8 = 64
constexpr size_t W3_SIZE = N_OUT * N_H2; // 1 * 8 = 8

// Total network parameters: 88 weights + 17 biases = 105 floats
typedef struct {
    // Weights (stored in flat, contiguous row-major order)
    float w1[W1_SIZE];
    float b1[N_H1];

    float w2[W2_SIZE];
    float b2[N_H2];

    float w3[W3_SIZE];
    float b3[N_OUT];
} NeuralNetwork;

// State buffer holding intermediate layer activations during forward inference
typedef struct {
    float h1[N_H1];
    float h2[N_H2];
    float out[N_OUT];
} NetworkActivations;

// ============================================================================
// 2. ACTIVATION FUNCTIONS & UTILITIES
// ============================================================================

// Standard file-scope helper function for generating random floats
static inline float random_weight(void) 
{
    // Generates a float between -0.5 and +0.5
    return ((float)rand() / (float)RAND_MAX) - 0.5f;
}

// ReLU: folds linear space (hinge function)
static inline float relu(float z) 
{
    return (z > 0.0f) ? z : 0.0f;
}

// Sigmoid: squashes final sum into a probability between 0.0 and 1.0
static inline float sigmoid(float z) 
{
    return 1.0f / (1.0f + expf(-z));
}

// ============================================================================
// 3. CORE COMPUTATION: DENSE LAYER
// ============================================================================
// Computes: out = Activation(weights * in + biases)
// Uses standard C 'restrict' to promise no pointer overlap for SIMD optimization.
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
        // Start with the node's bias threshold:
        float sum = biases[j];

        // Compute dot-product with incoming layer activations:
        const size_t row_offset = j * in_dim;
        for (size_t i = 0; i < in_dim; i++) {
            sum += in[i] * weights[row_offset + i];
        }

        // Apply non-linear activation:
        out[j] = use_sigmoid ? sigmoid(sum) : relu(sum);
    }
}

// Full Forward Pass from (x, y) input coordinates to final classification
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
// 4. WEIGHT INITIALIZATION
// ============================================================================
void init_network(NeuralNetwork *net) 
{
    // Seed standard PRNG for deterministic, reproducible initial weights
    srand(42);

    for (size_t i = 0; i < W1_SIZE; i++) net->w1[i] = random_weight();
    for (size_t i = 0; i < N_H1; i++)    net->b1[i] = 0.0f;

    for (size_t i = 0; i < W2_SIZE; i++) net->w2[i] = random_weight();
    for (size_t i = 0; i < N_H2; i++)    net->b2[i] = 0.0f;

    for (size_t i = 0; i < W3_SIZE; i++) net->w3[i] = random_weight();
    for (size_t i = 0; i < N_OUT; i++)   net->b3[i] = 0.0f;
}

// ============================================================================
// 5. TERMINAL VISUALIZER (The ASCII Grid)
// ============================================================================
void render_decision_boundary(const NeuralNetwork *net) 
{
    NetworkActivations acts = {}; // C23 empty zero-initialization

    constexpr int ROWS = 21;
    constexpr int COLS = 41;

    printf("\n--- CURRENT DECISION BOUNDARY (V1: Untrained / Random Weights) ---\n");
    for (int r = 0; r < ROWS; r++) {
        // Map row to y: +1.0 down to -1.0
        float y = 1.0f - (2.0f * (float)r / (float)(ROWS - 1));

        for (int c = 0; c < COLS; c++) {
            // Map column to x: -1.0 to +1.0
            float x = -1.0f + (2.0f * (float)c / (float)(COLS - 1));

            float input[N_IN] = { x, y };
            float prob = forward_pass(net, &acts, input);

            // Print '#' if predicted inside circle (>= 0.5), otherwise '.'
            if (prob >= 0.5f) {
                putchar('#');
            } else {
                putchar('.');
            }
        }
        putchar('\n');
    }
    printf("Legend: '#' = Network predicts INSIDE (>= 0.5), '.' = OUTSIDE (< 0.5)\n\n");
}

int main(void) 
{
    NeuralNetwork net = {}; // C23 zero-initialization
    init_network(&net);

    printf("Neural Network V1 Initialized:\n");
    printf("Total Parameters: %zu floats (%zu weights, %zu biases)\n", 
           W1_SIZE + W2_SIZE + W3_SIZE + N_H1 + N_H2 + N_OUT,
           W1_SIZE + W2_SIZE + W3_SIZE,
           N_H1 + N_H2 + N_OUT);

    // Test a single point at the origin (0.0, 0.0)
    NetworkActivations test_acts = {};
    float center_point[N_IN] = { 0.0f, 0.0f };
    float center_prob = forward_pass(&net, &test_acts, center_point);
    printf("Prediction at center (0.0, 0.0): %.4f (Probability)\n", center_prob);

    // Render the untrained field
    render_decision_boundary(&net);

    return 0;
}