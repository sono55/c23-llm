#include <stdio.h>

// In C23, an empty parameter list strictly means void (zero arguments)
int get_embedding_dim() 
{
    return 64;
}

int main(void) 
{
    // 1. Native keywords (no <stdbool.h> needed)
    bool is_initialized = true;
    float *weights = nullptr;

    // 2. Compile-time constant expression
    constexpr int hidden_layers = 4;

    // 3. Type inference
    auto dim = get_embedding_dim();

    // 4. Digit separators & binary literals
    long max_tokens = 1'000'000;
    unsigned char flag = 0b10100101;

    if (is_initialized && weights == nullptr) {
        printf("=========================================\n");
        printf(" C23 Environment Operational on Lunar Lake!\n");
        printf("=========================================\n");
        printf("Hidden Layers : %d\n", hidden_layers);
        printf("Embedding Dim : %d\n", dim);
        printf("Max Tokens    : %ld\n", max_tokens);
        printf("Bitmask       : 0x%X\n", flag);
    }

    return 0;
}