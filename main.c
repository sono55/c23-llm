#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ============================================================================
// 0. VERSION & TRAINING HYPERPARAMETERS (C23 constexpr)
// ============================================================================
constexpr int VERSION_MAJOR = 2;
constexpr int VERSION_MINOR = 0;

// Circle equation: x^2 + y^2 <= 0.5 (radius ~ 0.707)
constexpr float CIRCLE_RADIUS_SQ = 0.5f;

// Training settings (notice C23 digit separator in 200'000):
constexpr int   TRAIN_STEPS   = 200'000;
constexpr int   PRINT_EVERY   = 40'000;
constexpr float LEARNING_RATE = 0.05f;

// ============================================================================
// 1. TOPOLOGY SPECIFICATION (C23 constexpr)
// ============================================================================
constexpr size_t N_IN  = 2; // Input coordinates: (x, y)
constexpr size_t N_H1  = 8; // Hidden Layer 1: 8 neurons
constexpr size_t N_H2  = 8; // Hidden Layer 2: 8 neurons
constexpr size_t N_OUT = 1; // Output Layer: 1 neuron (probability)

// Matrix dimensions (row-major: [out_dim x in_dim]):
constexpr size_t W1_SIZE = N_H1 * N_IN;  // 8 * 2 = 16 weights
constexpr size_t W2_SIZE = N_H2 * N_H1;  // 8 * 8 = 64 weights
constexpr size_t W3_SIZE = N_OUT * N_H2; // 1 * 8 = 8 weights

// ----------------------------------------------------------------------------
// MEMORY SEPARATION ARCHITECTURE:
// 1. NeuralNetwork:      The 105 permanent knobs (weights & biases).
// 2. NetworkActivations: Signals flowing Left -> Right during Forward Pass.
// 3. NetworkErrors:      Error sensitivities flowing Right <- Left during
//                        Backward Pass (d1, d2, d3 in Diagram 2).
// ----------------------------------------------------------------------------
typedef struct {
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

typedef struct {
    float d_h1[N_H1];
    float d_h2[N_H2];
    float d_out[N_OUT];
} NetworkErrors;

// ============================================================================
// 2. ACTIVATION FUNCTIONS, DERIVATIVES & UTILITIES
// ============================================================================

// Generates small random float in [-0.5, +0.5] to break symmetry
static inline float random_weight(void) 
{
    return ((float)rand() / (float)RAND_MAX) - 0.5f;
}

// Generates random coordinate in [-1.0, +1.0] for training questions
static inline float random_coord(void) 
{
    return (((float)rand() / (float)RAND_MAX) * 2.0f) - 1.0f;
}

// ReLU Activation (Forward): f(z) = max(0, z)
static inline float relu(float z) 
{
    return (z > 0.0f) ? z : 0.0f;
}

// ReLU Derivative (Backward Slope):
// If the neuron fired (act > 0), slope is 1.0. If silent, slope is 0.0.
static inline float relu_slope(float act) 
{
    return (act > 0.0f) ? 1.0f : 0.0f;
}

// Sigmoid Activation (Forward): 1 / (1 + e^-z)
static inline float sigmoid(float z) 
{
    return 1.0f / (1.0f + expf(-z));
}

// Sigmoid Derivative (Backward Slope): out * (1.0 - out)
static inline float sigmoid_slope(float out) 
{
    return out * (1.0f - out);
}

// ============================================================================
// 3. FORWARD PASS (Left --> Right, Column by Column)
// ============================================================================
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
        float sum = biases[j];
        const size_t row_offset = j * in_dim;
        for (size_t i = 0; i < in_dim; i++) {
            sum += in[i] * weights[row_offset + i];
        }
        out[j] = use_sigmoid ? sigmoid(sum) : relu(sum);
    }
}

float forward_pass(const NeuralNetwork *restrict net, 
                   NetworkActivations *restrict acts, 
                   const float input[N_IN]) 
{
    dense_forward(input, acts->h1, net->w1, net->b1, N_IN, N_H1, false);
    dense_forward(acts->h1, acts->h2, net->w2, net->b2, N_H1, N_H2, false);
    dense_forward(acts->h2, acts->out, net->w3, net->b3, N_H2, N_OUT, true);

    return acts->out[0];
}

// ============================================================================
// 4. BACKWARD PASS (Right <-- Left, Column by Column) [NEW IN V2]
// ============================================================================

// Pulls error backwards from a destination layer (d_out) across the connecting
// weights into a source hidden layer (d_in), multiplying by the ReLU slope.
// Notice why we must finish all neurons 'j' in the right column first:
// each neuron 'i' in the left column sums blame from ALL neurons 'j' on the right!
static void dense_backward_relu(
    const float *restrict in_acts,  // Saved activations of source column
    float *restrict d_in,           // Output: calculated errors for source column
    const float *restrict d_out,    // Input: known errors from destination column
    const float *restrict weights,  // Weights connecting source -> destination
    size_t in_dim,
    size_t out_dim
) {
    for (size_t i = 0; i < in_dim; i++) {
        float error_sum = 0.0f;

        // Gather blame from every neuron 'j' in the layer to the right:
        for (size_t j = 0; j < out_dim; j++) {
            error_sum += d_out[j] * weights[j * in_dim + i];
        }

        // Chain Rule: multiply accumulated blame by this neuron's ReLU slope:
        d_in[i] = error_sum * relu_slope(in_acts[i]);
    }
}

// Diagram 1 - Step 4: Compute error sensitivities from Speaker back to Antenna
void backward_pass(const NeuralNetwork *restrict net,
                   const NetworkActivations *restrict acts,
                   NetworkErrors *restrict errs,
                   float target)
{
    // 1. Column 3 (Output Neuron):
    // Chain Rule Link 1 (out - target) * Link 2 (sigmoid_slope)
    float prob = acts->out[0];
    float raw_error = prob - target;
    errs->d_out[0] = raw_error * sigmoid_slope(prob);

    // 2. Column 2 (Hidden Layer 2): Pull error back across W3
    dense_backward_relu(acts->h2, errs->d_h2, errs->d_out, net->w3, N_H2, N_OUT);

    // 3. Column 1 (Hidden Layer 1): Pull error back across W2
    dense_backward_relu(acts->h1, errs->d_h1, errs->d_h2, net->w2, N_H1, N_H2);
}

// ============================================================================
// 5. GRADIENT DESCENT: TURN THE 105 KNOBS [NEW IN V2]
// ============================================================================

// Applies the Golden Rule to a single layer's weights and biases:
// Weight Gradient = (Destination Neuron Error) * (Incoming Wire Signal)
static void dense_update_weights(
    const float *restrict in_signals, // Signals that entered the wires
    const float *restrict d_out,      // Error at destination neurons
    float *restrict weights,          // Weights to adjust
    float *restrict biases,           // Biases to adjust
    size_t in_dim,
    size_t out_dim,
    float learning_rate
) {
    for (size_t j = 0; j < out_dim; j++) {
        // Bias has an implicit incoming signal of 1.0:
        biases[j] -= learning_rate * d_out[j];

        // Update each incoming wire weight for neuron 'j':
        const size_t row_offset = j * in_dim;
        for (size_t i = 0; i < in_dim; i++) {
            float gradient = d_out[j] * in_signals[i];
            weights[row_offset + i] -= learning_rate * gradient;
        }
    }
}

// Diagram 1 - Step 5: Adjust all weights now that all errors are known
void update_weights(NeuralNetwork *restrict net,
                    const NetworkActivations *restrict acts,
                    const NetworkErrors *restrict errs,
                    const float input[N_IN],
                    float learning_rate)
{
    // Tune Stage 1 (Input -> Hidden 1)
    dense_update_weights(input, errs->d_h1, net->w1, net->b1, N_IN, N_H1, learning_rate);

    // Tune Stage 2 (Hidden 1 -> Hidden 2)
    dense_update_weights(acts->h1, errs->d_h2, net->w2, net->b2, N_H1, N_H2, learning_rate);

    // Tune Stage 3 (Hidden 2 -> Output)
    dense_update_weights(acts->h2, errs->d_out, net->w3, net->b3, N_H2, N_OUT, learning_rate);
}

// ============================================================================
// 6. TRAINING ENGINE (Diagram 1 Outer Loop) [NEW IN V2]
// ============================================================================
void train_network(NeuralNetwork *net)
{
    NetworkActivations acts = {};
    NetworkErrors errs = {};

    printf("================================================================\n");
    printf("  STARTING TRAINING: %d Steps (Learning Rate: %.3f)\n", 
           TRAIN_STEPS, LEARNING_RATE);
    printf("================================================================\n");

    float running_loss = 0.0f;

    for (int step = 1; step <= TRAIN_STEPS; step++) {
        // STEP 1: Generate a random practice question (x, y) & Answer Key
        float x = random_coord();
        float y = random_coord();
        float input[N_IN] = { x, y };

        // Ground Truth: 1.0 if inside circle (x^2 + y^2 <= 0.5), else 0.0
        float target = ((x * x) + (y * y) <= CIRCLE_RADIUS_SQ) ? 1.0f : 0.0f;

        // STEP 2: Forward Pass (Left -> Right)
        float prob = forward_pass(net, &acts, input);

        // STEP 3: Measure Mean Squared Error Loss: 0.5 * (prob - target)^2
        float diff = prob - target;
        running_loss += 0.5f * (diff * diff);

        // STEP 4: Backward Pass (Right <- Left)
        backward_pass(net, &acts, &errs, target);

        // STEP 5: Turn the 105 Knobs
        update_weights(net, &acts, &errs, input, LEARNING_RATE);

        // Print progress every PRINT_EVERY steps:
        if (step % PRINT_EVERY == 0) {
            printf("  Step %6d / %d | Average Loss (Error): %.5f\n", 
                   step, TRAIN_STEPS, running_loss / (float)PRINT_EVERY);
            running_loss = 0.0f;
        }
    }
    printf("================================================================\n\n");
}

// ============================================================================
// 7. INITIALIZATION, INSPECTION & VISUALIZATION
// ============================================================================
void init_network(NeuralNetwork *net) 
{
    srand(42); // Deterministic seed

    for (size_t i = 0; i < W1_SIZE; i++) net->w1[i] = random_weight();
    for (size_t i = 0; i < N_H1; i++)    net->b1[i] = 0.0f;

    for (size_t i = 0; i < W2_SIZE; i++) net->w2[i] = random_weight();
    for (size_t i = 0; i < N_H2; i++)    net->b2[i] = 0.0f;

    for (size_t i = 0; i < W3_SIZE; i++) net->w3[i] = random_weight();
    for (size_t i = 0; i < N_OUT; i++)   net->b3[i] = 0.0f;
}

void print_network_parameters(const NeuralNetwork *net, const char *title) 
{
    printf("\n================================================================\n");
    printf("  %s\n", title);
    printf("================================================================\n");

    printf("\n[Layer 1: Input (2) -> Hidden 1 (8)]\n");
    for (size_t j = 0; j < N_H1; j++) {
        printf("  Node %zu | Bias: %+6.3f | Weights: [ ", j, net->b1[j]);
        for (size_t i = 0; i < N_IN; i++) {
            printf("%+6.3f ", net->w1[j * N_IN + i]);
        }
        printf("]\n");
    }

    printf("\n[Layer 2: Hidden 1 (8) -> Hidden 2 (8)]\n");
    for (size_t j = 0; j < N_H2; j++) {
        printf("  Node %zu | Bias: %+6.3f | Weights: [ ", j, net->b2[j]);
        for (size_t i = 0; i < N_H1; i++) {
            printf("%+6.3f ", net->w2[j * N_H1 + i]);
        }
        printf("]\n");
    }

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

void render_decision_boundary(const NeuralNetwork *net, const char *label) 
{
    NetworkActivations acts = {};

    constexpr int ROWS = 21;
    constexpr int COLS = 41;

    printf("--- DECISION BOUNDARY: %s (V%d.%d) ---\n", label, VERSION_MAJOR, VERSION_MINOR);
    for (int r = 0; r < ROWS; r++) {
        float y = 1.0f - (2.0f * (float)r / (float)(ROWS - 1));

        for (int c = 0; c < COLS; c++) {
            float x = -1.0f + (2.0f * (float)c / (float)(COLS - 1));

            float input[N_IN] = { x, y };
            float prob = forward_pass(net, &acts, input);

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
    NeuralNetwork net = {};
    init_network(&net);

    printf("Neural Network V%d.%d Initialized:\n", VERSION_MAJOR, VERSION_MINOR);
    printf("Total Parameters: %zu floats (%zu weights, %zu biases)\n", 
           W1_SIZE + W2_SIZE + W3_SIZE + N_H1 + N_H2 + N_OUT,
           W1_SIZE + W2_SIZE + W3_SIZE,
           N_H1 + N_H2 + N_OUT);

    // 1. Inspect initial untrained state:
    print_network_parameters(&net, "INITIAL UNTRAINED PARAMETERS");
    render_decision_boundary(&net, "BEFORE TRAINING");

    // 2. Run Backpropagation Training Loop:
    train_network(&net);

    // 3. Inspect final trained state:
    print_network_parameters(&net, "FINAL TRAINED PARAMETERS");
    render_decision_boundary(&net, "AFTER TRAINING");

    // 4. Verify specific test coordinates:
    NetworkActivations test_acts = {};
    float center_pt[N_IN] = { 0.0f, 0.0f }; // Inside (0^2 + 0^2 = 0.0 <= 0.5)
    float corner_pt[N_IN] = { 0.9f, 0.9f }; // Outside (0.9^2 + 0.9^2 = 1.62 > 0.5)

    printf("Verification Queries:\n");
    printf("  Center (0.0, 0.0) [Expected ~1.0]: %.4f\n", forward_pass(&net, &test_acts, center_pt));
    printf("  Corner (0.9, 0.9) [Expected ~0.0]: %.4f\n", forward_pass(&net, &test_acts, corner_pt));

    return 0;
}