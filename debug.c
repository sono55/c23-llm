#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ============================================================================
// 0. COMPILE-TIME DEBUGGER SWITCH & VERSION (C23 constexpr)
// ============================================================================
// Set STEP_DEBUG to 1 for interactive stepping, or 0 for full-speed batch run.
// Can also be overridden at compile time via: gcc -DSTEP_DEBUG=0 ...
#ifndef STEP_DEBUG
#define STEP_DEBUG 1
#endif

#if STEP_DEBUG
#include <termios.h>
#include <unistd.h>
#endif

constexpr int VERSION_MAJOR = 2;
constexpr int VERSION_MINOR = 1;

// Circle equation: x^2 + y^2 <= 0.5 (radius ~ 0.707)
constexpr float CIRCLE_RADIUS_SQ = 0.5f;

// Training hyperparameters:
constexpr int   TRAIN_STEPS   = 200'000;
constexpr int   PRINT_EVERY   = 40'000;
constexpr int   FAST_FWD_STEP = 5'000;
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

static inline float random_weight(void) 
{
    return ((float)rand() / (float)RAND_MAX) - 0.5f;
}

static inline float random_coord(void) 
{
    return (((float)rand() / (float)RAND_MAX) * 2.0f) - 1.0f;
}

static inline float relu(float z) 
{
    return (z > 0.0f) ? z : 0.0f;
}

// ReLU Derivative: 1.0 if neuron fired (> 0), 0.0 if silent
static inline float relu_slope(float act) 
{
    return (act > 0.0f) ? 1.0f : 0.0f;
}

static inline float sigmoid(float z) 
{
    return 1.0f / (1.0f + expf(-z));
}

// Sigmoid Derivative: out * (1.0 - out)
static inline float sigmoid_slope(float out) 
{
    return out * (1.0f - out);
}

#if STEP_DEBUG
// Reads a single keypress immediately without waiting for Enter (POSIX termios)
static char read_single_key(void) 
{
    struct termios oldt = {};
    struct termios newt = {};
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= (tcflag_t)~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);

    char ch = 0;
    if (read(STDIN_FILENO, &ch, 1) < 0) {
        ch = 'q';
    }

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    return ch;
}
#endif

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
// 4. BACKWARD PASS (Right <-- Left, Column by Column)
// ============================================================================
static void dense_backward_relu(
    const float *restrict in_acts,
    float *restrict d_in,
    const float *restrict d_out,
    const float *restrict weights,
    size_t in_dim,
    size_t out_dim
) {
    for (size_t i = 0; i < in_dim; i++) {
        float error_sum = 0.0f;
        for (size_t j = 0; j < out_dim; j++) {
            error_sum += d_out[j] * weights[j * in_dim + i];
        }
        d_in[i] = error_sum * relu_slope(in_acts[i]);
    }
}

void backward_pass(const NeuralNetwork *restrict net,
                   const NetworkActivations *restrict acts,
                   NetworkErrors *restrict errs,
                   float target)
{
    float prob = acts->out[0];
    float raw_error = prob - target;
    errs->d_out[0] = raw_error * sigmoid_slope(prob);

    dense_backward_relu(acts->h2, errs->d_h2, errs->d_out, net->w3, N_H2, N_OUT);
    dense_backward_relu(acts->h1, errs->d_h1, errs->d_h2, net->w2, N_H1, N_H2);
}

// ============================================================================
// 5. GRADIENT DESCENT: TURN THE 105 KNOBS
// ============================================================================
static void dense_update_weights(
    const float *restrict in_signals,
    const float *restrict d_out,
    float *restrict weights,
    float *restrict biases,
    size_t in_dim,
    size_t out_dim,
    float learning_rate
) {
    for (size_t j = 0; j < out_dim; j++) {
        biases[j] -= learning_rate * d_out[j];

        const size_t row_offset = j * in_dim;
        for (size_t i = 0; i < in_dim; i++) {
            float gradient = d_out[j] * in_signals[i];
            weights[row_offset + i] -= learning_rate * gradient;
        }
    }
}

void update_weights(NeuralNetwork *restrict net,
                    const NetworkActivations *restrict acts,
                    const NetworkErrors *restrict errs,
                    const float input[N_IN],
                    float learning_rate)
{
    dense_update_weights(input, errs->d_h1, net->w1, net->b1, N_IN, N_H1, learning_rate);
    dense_update_weights(acts->h1, errs->d_h2, net->w2, net->b2, N_H1, N_H2, learning_rate);
    dense_update_weights(acts->h2, errs->d_out, net->w3, net->b3, N_H2, N_OUT, learning_rate);
}

// ============================================================================
// 6. INSPECTION & VISUALIZATION (5-Decimal Precision for Stepping)
// ============================================================================
void print_network_parameters(const NeuralNetwork *net, const char *title) 
{
    printf("========================================================================================\n");
    printf("  %s (V%d.%d)\n", title, VERSION_MAJOR, VERSION_MINOR);
    printf("========================================================================================\n");

    // Layer 1 printed 2 nodes per line for compact dashboard viewing:
    printf("[Layer 1: Input (2) -> Hidden 1 (8)]\n");
    for (size_t j = 0; j < N_H1; j += 2) {
        printf("  N%zu: b=%+8.5f w=[%+8.5f, %+8.5f]   |   N%zu: b=%+8.5f w=[%+8.5f, %+8.5f]\n",
               j,     net->b1[j],     net->w1[j * N_IN + 0],     net->w1[j * N_IN + 1],
               j + 1, net->b1[j + 1], net->w1[(j + 1) * N_IN + 0], net->w1[(j + 1) * N_IN + 1]);
    }

    printf("\n[Layer 2: Hidden 1 (8) -> Hidden 2 (8)]\n");
    for (size_t j = 0; j < N_H2; j++) {
        printf("  N%zu: b=%+8.5f w=[ ", j, net->b2[j]);
        for (size_t i = 0; i < N_H1; i++) {
            printf("%+8.5f ", net->w2[j * N_H1 + i]);
        }
        printf("]\n");
    }

    printf("\n[Layer 3: Hidden 2 (8) -> Output (1)]\n");
    for (size_t j = 0; j < N_OUT; j++) {
        printf("  N%zu: b=%+8.5f w=[ ", j, net->b3[j]);
        for (size_t i = 0; i < N_H2; i++) {
            printf("%+8.5f ", net->w3[j * N_H2 + i]);
        }
        printf("]\n");
    }
}

void render_decision_boundary(const NeuralNetwork *net, const char *label) 
{
    NetworkActivations acts = {};

    constexpr int ROWS = 17;
    constexpr int COLS = 37;

    printf("\n--- DECISION BOUNDARY: %s ---\n", label);
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
}

// ============================================================================
// 7. TRAINING ENGINE (With Interactive Single-Step Debugger)
// ============================================================================
void train_network(NeuralNetwork *net)
{
    NetworkActivations acts = {};
    NetworkErrors errs = {};

#if STEP_DEBUG
    bool interactive = true;
    int next_pause_step = 1; // Pause immediately on Step 1
#endif

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
        float step_loss = 0.5f * (diff * diff);
        running_loss += step_loss;

        // STEP 4: Backward Pass (Right <- Left)
        backward_pass(net, &acts, &errs, target);

        // STEP 5: Turn the 105 Knobs
        update_weights(net, &acts, &errs, input, LEARNING_RATE);

#if STEP_DEBUG
        if (interactive && step >= next_pause_step) {
            // Clear terminal & home cursor for in-place animation:
            printf("\033[2J\033[H");

            char banner[64];
            snprintf(banner, sizeof(banner), "LIVE DEBUGGER - AFTER PASS %d / %d", step, TRAIN_STEPS);
            print_network_parameters(net, banner);
            render_decision_boundary(net, "LIVE FIELD VIEW");

            printf("\n----------------------------------------------------------------------------------------\n");
            printf("  Last Pass #%d | Coord: (x=%+6.3f, y=%+6.3f) | Target: %.1f | Guess: %.5f | Loss: %.5f\n",
                   step, x, y, target, prob, step_loss);
            printf("  CONTROLS: [SPACE] = +1 Pass  |  [F] = Fast-Forward +%d Passes  |  [Q] = Run to End\n",
                   FAST_FWD_STEP);
            printf("----------------------------------------------------------------------------------------\n");
            fflush(stdout);

            while (true) {
                char key = read_single_key();
                if (key == ' ') {
                    next_pause_step = step + 1;
                    break;
                } else if (key == 'f' || key == 'F') {
                    next_pause_step = step + FAST_FWD_STEP;
                    if (next_pause_step > TRAIN_STEPS) {
                        next_pause_step = TRAIN_STEPS;
                    }
                    break;
                } else if (key == 'q' || key == 'Q') {
                    interactive = false;
                    printf("\nResuming full-speed training...\n");
                    break;
                }
            }
        }
#endif

        // Standard progress printout when not in interactive mode:
        if (step % PRINT_EVERY == 0) {
#if STEP_DEBUG
            if (!interactive)
#endif
            {
                printf("  Step %6d / %d | Average Loss (Error): %.5f\n", 
                       step, TRAIN_STEPS, running_loss / (float)PRINT_EVERY);
            }
            running_loss = 0.0f;
        }
    }
}

void init_network(NeuralNetwork *net) 
{
    srand(42);

    for (size_t i = 0; i < W1_SIZE; i++) net->w1[i] = random_weight();
    for (size_t i = 0; i < N_H1; i++)    net->b1[i] = 0.0f;

    for (size_t i = 0; i < W2_SIZE; i++) net->w2[i] = random_weight();
    for (size_t i = 0; i < N_H2; i++)    net->b2[i] = 0.0f;

    for (size_t i = 0; i < W3_SIZE; i++) net->w3[i] = random_weight();
    for (size_t i = 0; i < N_OUT; i++)   net->b3[i] = 0.0f;
}

int main(void) 
{
    NeuralNetwork net = {};
    init_network(&net);

#if STEP_DEBUG
    printf("\033[2J\033[H");
    print_network_parameters(&net, "PASS 0 / 200000 (INITIAL UNTRAINED STATE)");
    render_decision_boundary(&net, "INITIAL FIELD VIEW");
    printf("\n----------------------------------------------------------------------------------------\n");
    printf("  DEBUGGER READY: Inspect initial weights (all biases = 0.00000).\n");
    printf("  Press [SPACE] to run Pass #1, [F] to jump +5,000 passes, or [Q] for full speed.\n");
    printf("----------------------------------------------------------------------------------------\n");
    fflush(stdout);
    char start_key = read_single_key();
    if (start_key == 'q' || start_key == 'Q') {
        // User chose still to run, handled inside train_network
    }
#else
    printf("Neural Network V%d.%d Initialized:\n", VERSION_MAJOR, VERSION_MINOR);
    print_network_parameters(&net, "INITIAL UNTRAINED PARAMETERS");
    render_decision_boundary(&net, "BEFORE TRAINING");
#endif

    // Run Backpropagation Training Loop:
    train_network(&net);

    // Final summary:
    print_network_parameters(&net, "FINAL TRAINED PARAMETERS");
    render_decision_boundary(&net, "AFTER TRAINING");

    NetworkActivations test_acts = {};
    float center_pt[N_IN] = { 0.0f, 0.0f };
    float corner_pt[N_IN] = { 0.9f, 0.9f };

    printf("\nVerification Queries:\n");
    printf("  Center (0.0, 0.0) [Expected ~1.0]: %.4f\n", forward_pass(&net, &test_acts, center_pt));
    printf("  Corner (0.9, 0.9) [Expected ~0.0]: %.4f\n", forward_pass(&net, &test_acts, corner_pt));

    return 0;
}