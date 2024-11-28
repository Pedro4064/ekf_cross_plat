#include "arm_math.h"
#include <stdio.h>

int main() {
    float32_t A[2] = {1, 1};  // 2x2 matrix
    float32_t B[2] = {1, 1};  // 2x2 matrix
    float32_t C[4];                 // Result 2x2 matrix
    arm_matrix_instance_f32 matA, matB, matC;

    arm_mat_init_f32(&matA, 2, 1, (float32_t *)A);
    arm_mat_init_f32(&matB, 1, 2, B);
    arm_mat_init_f32(&matC, 2, 2, C);

    if (arm_mat_mult_f32(&matA, &matB, &matC) == ARM_MATH_SUCCESS) {
        printf("Matrix multiplication successful! Result:\n");
        for (int i = 0; i < 4; i++) {
            printf("%f ", C[i]);
        }
    } else {
        printf("Matrix multiplication failed.\n");
    }

    return 0;
}