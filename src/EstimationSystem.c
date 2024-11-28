#include "EstimationSystem.h"


float32_t fStateVector[8];

float32_t fZ[2][1];
float32_t fU[2][1];
float32_t fX_hat_priori[NUM_STATES];
float32_t fP_priori[NUM_STATES][NUM_STATES];
float32_t fK[NUM_STATES][2];
float32_t fF_jacobian[NUM_STATES][NUM_STATES];
float32_t fH_jacobian[2][NUM_STATES];
float32_t fH_nonlinear[2];

float32_t fP_posteriori[NUM_STATES][NUM_STATES] = {
                                                            {0.1, 0, 0, 0, 0, 0, 0, 0},
                                                            {0, 0.1, 0, 0, 0, 0, 0, 0},
                                                            {0, 0, 0.1, 0, 0, 0, 0, 0},
                                                            {0, 0, 0, 0.1, 0, 0, 0, 0},
                                                            {0, 0, 0, 0, 0.1, 0, 0, 0},
                                                            {0, 0, 0, 0, 0, 0.1, 0, 0},
                                                            {0, 0, 0, 0, 0, 0, 0.1, 0},
                                                            {0, 0, 0, 0, 0, 0, 0, 0.1},
                                                        };

float32_t fQ[NUM_STATES][NUM_STATES] = {
                                                            {0.1, 0, 0, 0, 0, 0, 0, 0},
                                                            {0, 0.1, 0, 0, 0, 0, 0, 0},
                                                            {0, 0, 0.1, 0, 0, 0, 0, 0},
                                                            {0, 0, 0, 0.1, 0, 0, 0, 0},
                                                            {0, 0, 0, 0, 0.1, 0, 0, 0},
                                                            {0, 0, 0, 0, 0, 0.1, 0, 0},
                                                            {0, 0, 0, 0, 0, 0, 0.1, 0},
                                                            {0, 0, 0, 0, 0, 0, 0, 0.1},
                                                        };

float32_t fR[2][2] = {
                                {10.0f, 0.0f},
                                {0.0f, 10.00f}
                            };





static arm_matrix_instance_f32 mZ;
static arm_matrix_instance_f32 mU;
static arm_matrix_instance_f32 mX_hat_posteriori;
static arm_matrix_instance_f32 mX_hat_priori;
static arm_matrix_instance_f32 mP_posteriori;
static arm_matrix_instance_f32 mP_priori;
static arm_matrix_instance_f32 mK;
static arm_matrix_instance_f32 mH_jacobian;
static arm_matrix_instance_f32 mH_nonlinear;
static arm_matrix_instance_f32 mF_jacobian;
static arm_matrix_instance_f32 mQ;
static arm_matrix_instance_f32 mR;



// void vEstimationSystemInit(SystemState* pSysState, TelemetryData* pTelemData, MotorCommands* pMotorCommands){
//     // Save necessary system variables
//     pSystemState = pSysState;
//     pTelemetryData = pTelemData;
//     pMotorCommands = pMotorCommands;

//     // Initialize all matrices used in the extended kalman filter implementation
//     arm_mat_init_f32(&mZ, 2, 1, &fZ[0][0]);
//     arm_mat_init_f32(&mU, 2, 1, &fU[0][0]);
//     arm_mat_init_f32(&mX_hat_posteriori, NUM_STATES, 1, &(fStateVector[0]));
//     arm_mat_init_f32(&mX_hat_priori, NUM_STATES, 1, &fX_hat_priori[0]);
//     arm_mat_init_f32(&mP_posteriori, NUM_STATES, NUM_STATES, &fP_posteriori[0][0]);
//     arm_mat_init_f32(&mP_priori, NUM_STATES, NUM_STATES, &fP_priori[0][0]);
//     arm_mat_init_f32(&mQ, NUM_STATES, NUM_STATES, &fQ[0][0]);
//     arm_mat_init_f32(&mR, 2, 2, &fR[0][0]);
//     arm_mat_init_f32(&mK, NUM_STATES, 2, &fK[0][0]);
//     arm_mat_init_f32(&mF_jacobian, NUM_STATES, NUM_STATES, &fF_jacobian[0][0]);
//     arm_mat_init_f32(&mH_jacobian, 2, NUM_STATES, &fH_jacobian[0][0]);
//     arm_mat_init_f32(&mH_nonlinear, 2, 1, &fH_nonlinear[0]);

// }

/**
 * @brief Computes the dynamic model transition matrix f(x, u) and its jacobian F(x, u)
 * 
 */
void vEstimationSystemComputeDynamicModel(float32_t u1, float32_t u2){
    float32_t Ts = 0.1f;
    float32_t r = 0.035f;
    float32_t L = 0.1f;

//    float32_t u[2] = {pMotorCommands->fLeftMotorSpeed, pMotorCommands->fRightMotorSpeed};
    float32_t u[2] = {u1, u2};

    // X Hat priori calculations
    fX_hat_priori[0] = fStateVector[0] + fStateVector[1] * Ts + 0.5f * fStateVector[2] * Ts * Ts;
    fX_hat_priori[1] = (r / 2.0f) * cosf(fStateVector[6]) * (u[0] + u[1]);
    fX_hat_priori[2] = fStateVector[2];
    fX_hat_priori[3] = fStateVector[3] + fStateVector[4] * Ts + 0.5f * fStateVector[5] * Ts * Ts;
    fX_hat_priori[4] = (r / 2.0f) * sinf(fStateVector[6]) * (u[0] + u[1]);
    fX_hat_priori[5] = fStateVector[5];
    fX_hat_priori[6] = fStateVector[6] + fStateVector[7] * Ts;
    fX_hat_priori[7] = (r / L) * (u[1] - u[0]);


    // Compute transition matrix jacobian
    float32_t *x = fStateVector;

    fF_jacobian[0][0] = 1.0f;
    fF_jacobian[0][1] = Ts;
    fF_jacobian[0][2] = 0.5f * Ts * Ts;

    fF_jacobian[1][6] = -(r / 2.0f) * sinf(x[6]) * (u[0] + u[1]);

    fF_jacobian[3][3] = 1.0f;
    fF_jacobian[3][4] = Ts;
    fF_jacobian[3][5] = 0.5f * Ts * Ts;

    fF_jacobian[4][6] = (r / 2.0f) * cosf(x[6]) * (u[0] + u[1]);

    fF_jacobian[6][6] = 1.0f;
    fF_jacobian[6][7] = Ts;

    fF_jacobian[7][7] = 0.0f; // Add other terms as required


}

/**
 * @brief Computes both the nonlinear observation matrix h(x, u) and its jacobian H(x, u)
 * 
 */
void vEstimationSystemComputeObservation(void){

    // Compute Jacobian of H
    float32_t *x = fX_hat_priori;

    float32_t t2 = cosf(x[6]); // x(7) in MATLAB, zero-based index 6 in C
    float32_t t3 = sinf(x[6]);

    fH_jacobian[0][4] = t2; // Corresponds to t2
    fH_jacobian[1][4] = t3; // Corresponds to t3
    fH_jacobian[1][2] = -t3 * x[2] + t2 * x[5]; // -t3 * x(3) + t2 * x(6) in MATLAB
    fH_jacobian[1][6] = 1.0f; // Corresponds to 1.0

    // Computes the nonlinear observation matrix h
    fH_nonlinear[0] = x[2] * cosf(x[6]) + x[5] * sinf(x[6]); // x(3)*cos(x(7)) + x(6)*sin(x(7))
    fH_nonlinear[1] = x[7]; // x(8)



}

void vEstimationSystemEkfPredict(void){


    float32_t fF_trans[NUM_STATES][NUM_STATES]; // Transpose of F
    float32_t fF_P[NUM_STATES][NUM_STATES];     // Intermediate result F * P_posteriori
    float32_t fF_P_Ft[NUM_STATES][NUM_STATES];  // Intermediate result F * P_posteriori * F^T

    arm_matrix_instance_f32 mF_trans, mF_P, mF_P_Ft;
    arm_mat_init_f32(&mF_trans, NUM_STATES, NUM_STATES, &fF_trans[0][0]);
    arm_mat_init_f32(&mF_P, NUM_STATES, NUM_STATES, &fF_P[0][0]);
    arm_mat_init_f32(&mF_P_Ft, NUM_STATES, NUM_STATES, &fF_P_Ft[0][0]);

    // Compute F * P_posteriori
    arm_mat_mult_f32(&mF_jacobian, &mP_posteriori, &mF_P);

    // Compute F^T
    arm_mat_trans_f32(&mF_jacobian, &mF_trans);

    // Compute F * P_posteriori * F^T
    arm_mat_mult_f32(&mF_P, &mF_trans, &mF_P_Ft);

    // Add Q to get P_priori
    arm_mat_add_f32(&mF_P_Ft, &mQ, &mP_priori);

}

void vEstimationSYstemCalculateK(void){
    float32_t fH_trans[NUM_STATES][2];  // Transpose of H
    float32_t fHPH_t_R[2][2];          // H * P_priori * H' + R
    float32_t fHPH_t_R_inv[2][2];      // Inverse of H * P_priori * H' + R
    float32_t fK_temp[NUM_STATES][2];  // K intermediate
    float32_t fHPH_t[2][2];            // H * P_priori * H'

    arm_matrix_instance_f32 mH_trans;
    arm_matrix_instance_f32 mHPH_t;
    arm_matrix_instance_f32 mHPH_t_R;
    arm_matrix_instance_f32 mHPH_t_R_inv;
    arm_matrix_instance_f32 mK_temp;

    static char cFirstIteration = 1;

    if(cFirstIteration){
        // Initialize all matrices used in the extended kalman filter implementation
        arm_mat_init_f32(&mP_priori, NUM_STATES, NUM_STATES, &fP_priori[0][0]);
        arm_mat_init_f32(&mR, 2, 2, &fR[0][0]);
        arm_mat_init_f32(&mK, NUM_STATES, 2, &fK[0][0]);
        arm_mat_init_f32(&mH_jacobian, 2, NUM_STATES, &fH_jacobian[0][0]);

        cFirstIteration = 0;
        return;
    }

    arm_mat_init_f32(&mH_trans, NUM_STATES, 2, (float32_t *)fH_trans);
    arm_mat_init_f32(&mHPH_t_R, 2, 2, (float32_t *)fHPH_t_R);
    arm_mat_init_f32(&mHPH_t_R_inv, 2, 2, (float32_t *)fHPH_t_R_inv);
    arm_mat_init_f32(&mK_temp, NUM_STATES, 2, (float32_t *)fK_temp);
    arm_mat_init_f32(&mHPH_t, 2, 2, &fHPH_t[0][0]);

    // Calculate K gain //////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////
    // H' (Transpose of H_jacobian)
    arm_mat_trans_f32(&mH_jacobian, &mH_trans);

    // H * P_priori * H'
    arm_mat_mult_f32(&mH_jacobian, &mP_priori, &mHPH_t);
    arm_mat_mult_f32(&mHPH_t, &mH_trans, &mHPH_t);

    // H * P_priori * H' + R
    arm_mat_add_f32(&mHPH_t, &mR, &mHPH_t_R);

    // Inverse of H * P_priori * H' + R
    arm_mat_inverse_f32(&mHPH_t_R, &mHPH_t_R_inv);

    // P_priori * H'
    arm_mat_mult_f32(&mP_priori, &mH_trans, &mK_temp);

    // K = P_priori * H' * inv(H * P_priori * H' + R)
    arm_mat_mult_f32(&mK_temp, &mHPH_t_R_inv, &mK);


}

void vEstimationSystemCalculateXposteriori(float32_t z1, float32_t z2){
    static char cFirstIteration = 1;
    if(cFirstIteration){
        // Initialize all matrices used in the extended kalman filter implementation
        arm_mat_init_f32(&mX_hat_posteriori, NUM_STATES, 1, &(fStateVector[0]));
        arm_mat_init_f32(&mX_hat_priori, NUM_STATES, 1, &fX_hat_priori[0]);
        arm_mat_init_f32(&mK, NUM_STATES, 2, &fK[0][0]);

        cFirstIteration = 0;
        return;
    }

    fZ[0][0] = z1;
    fZ[1][0] = z2;

    float32_t fZ_h[2][1];  // z - h
    float32_t fKZ_h[NUM_STATES][1]; // K * (z - h)

    arm_matrix_instance_f32 mZ_h;
    arm_matrix_instance_f32 mKZ_h;

    arm_mat_init_f32(&mZ_h, 2, 1, (float32_t *)fZ_h);
    arm_mat_init_f32(&mKZ_h, NUM_STATES, 1, (float32_t *)fKZ_h);

    // z - h
    for (int i = 0; i < 2; i++) {
        fZ_h[i][0] = fZ[i][0] - fH_nonlinear[i];
    }

    // K * (z - h)
    arm_mat_mult_f32(&mK, &mZ_h, &mKZ_h);

    // x_hat_posteriori = x_hat_priori + K * (z - h)
    arm_mat_add_f32(&mX_hat_priori, &mKZ_h, &mX_hat_posteriori);
}

void vEstimationSystemCalculatePposteriori(){

    static char cFirstIteration = 1;
    if(cFirstIteration){
        // Initialize all matrices used in the extended kalman filter implementation
        arm_mat_init_f32(&mP_posteriori, NUM_STATES, NUM_STATES, &fP_posteriori[0][0]);
        arm_mat_init_f32(&mP_priori, NUM_STATES, NUM_STATES, &fP_priori[0][0]);
        arm_mat_init_f32(&mR, 2, 2, &fR[0][0]);
        arm_mat_init_f32(&mK, NUM_STATES, 2, &fK[0][0]);
        arm_mat_init_f32(&mH_jacobian, 2, NUM_STATES, &fH_jacobian[0][0]);

        cFirstIteration = 0;
        return;
    }
    // Calculate P   /////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////
    float32_t fI[NUM_STATES][NUM_STATES];  // Identity matrix
    float32_t fIKH[NUM_STATES][NUM_STATES]; // (I - K * H)
    float32_t fIKH_t[NUM_STATES][NUM_STATES]; // Transpose of (I - K * H)
    float32_t fIKHP[NUM_STATES][NUM_STATES]; // (I - K * H) * P_priori
    float32_t fKRK[NUM_STATES][NUM_STATES]; // K * R * K'

    arm_matrix_instance_f32 mI;
    arm_matrix_instance_f32 mIKH;
    arm_matrix_instance_f32 mIKH_t;
    arm_matrix_instance_f32 mIKHP;
    arm_matrix_instance_f32 mKRK;

    arm_mat_init_f32(&mI, NUM_STATES, NUM_STATES, (float32_t *)fI);
    arm_mat_init_f32(&mIKH, NUM_STATES, NUM_STATES, (float32_t *)fIKH);
    arm_mat_init_f32(&mIKH_t, NUM_STATES, NUM_STATES, (float32_t *)fIKH_t);
    arm_mat_init_f32(&mIKHP, NUM_STATES, NUM_STATES, (float32_t *)fIKHP);
    arm_mat_init_f32(&mKRK, NUM_STATES, NUM_STATES, (float32_t *)fKRK);

    // Identity matrix
    for (int i = 0; i < NUM_STATES; i++) {
        for (int j = 0; j < NUM_STATES; j++) {
            fI[i][j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    // I - K * H
    arm_mat_mult_f32(&mK, &mH_jacobian, &mIKH);
    arm_mat_sub_f32(&mI, &mIKH, &mIKH);

    // (I - K * H)' (Transpose)
    arm_mat_trans_f32(&mIKH, &mIKH_t);

    // (I - K * H) * P_priori * (I - K * H)'
    arm_mat_mult_f32(&mIKH, &mP_priori, &mIKHP);
    arm_mat_mult_f32(&mIKHP, &mIKH_t, &mP_posteriori);

    // K * R * K'
    arm_mat_mult_f32(&mK, &mR, &mKRK);
    arm_mat_mult_f32(&mKRK, &mK, &mKRK);

    // Add K * R * K' to P_posteriori
    arm_mat_add_f32(&mP_posteriori, &mKRK, &mP_posteriori);
}

void vEstimationSystemEkfEstimate(float32_t z1, float32_t z2){

     static char cFirstIteration = 1;

    if(cFirstIteration){
        // Initialize all matrices used in the extended kalman filter implementation
        arm_mat_init_f32(&mZ, 2, 1, &fZ[0][0]);
        arm_mat_init_f32(&mU, 2, 1, &fU[0][0]);
        arm_mat_init_f32(&mX_hat_posteriori, NUM_STATES, 1, &(fStateVector[0]));
        arm_mat_init_f32(&mX_hat_priori, NUM_STATES, 1, &fX_hat_priori[0]);
        arm_mat_init_f32(&mP_posteriori, NUM_STATES, NUM_STATES, &fP_posteriori[0][0]);
        arm_mat_init_f32(&mP_priori, NUM_STATES, NUM_STATES, &fP_priori[0][0]);
        arm_mat_init_f32(&mQ, NUM_STATES, NUM_STATES, &fQ[0][0]);
        arm_mat_init_f32(&mR, 2, 2, &fR[0][0]);
        arm_mat_init_f32(&mK, NUM_STATES, 2, &fK[0][0]);
        arm_mat_init_f32(&mF_jacobian, NUM_STATES, NUM_STATES, &fF_jacobian[0][0]);
        arm_mat_init_f32(&mH_jacobian, 2, NUM_STATES, &fH_jacobian[0][0]);
        arm_mat_init_f32(&mH_nonlinear, 2, 1, &fH_nonlinear[0]);

        cFirstIteration = 0;
        return;
    }

    float32_t fH_trans[NUM_STATES][2];  // Transpose of H
    float32_t fHPH_t[2][2];            // H * P_priori * H'
    float32_t fHPH_t_R[2][2];          // H * P_priori * H' + R
    float32_t fHPH_t_R_inv[2][2];      // Inverse of H * P_priori * H' + R
    float32_t fK_temp[NUM_STATES][2];  // K intermediate

    arm_matrix_instance_f32 mH_trans;
    arm_matrix_instance_f32 mHPH_t;
    arm_matrix_instance_f32 mHPH_t_R;
    arm_matrix_instance_f32 mHPH_t_R_inv;
    arm_matrix_instance_f32 mK_temp;

    arm_mat_init_f32(&mH_trans, NUM_STATES, 2, (float32_t *)fH_trans);
    arm_mat_init_f32(&mHPH_t, 2, 2, (float32_t *)fHPH_t);
    arm_mat_init_f32(&mHPH_t_R, 2, 2, (float32_t *)fHPH_t_R);
    arm_mat_init_f32(&mHPH_t_R_inv, 2, 2, (float32_t *)fHPH_t_R_inv);
    arm_mat_init_f32(&mK_temp, NUM_STATES, 2, (float32_t *)fK_temp);

    // Calculate K gain //////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////
    // H' (Transpose of H_jacobian)
    arm_mat_trans_f32(&mH_jacobian, &mH_trans);

    // H * P_priori * H'
    arm_mat_mult_f32(&mH_jacobian, &mP_priori, &mHPH_t);
    arm_mat_mult_f32(&mHPH_t, &mH_trans, &mHPH_t);

    // H * P_priori * H' + R
    arm_mat_add_f32(&mHPH_t, &mR, &mHPH_t_R);

    // Inverse of H * P_priori * H' + R
    arm_mat_inverse_f32(&mHPH_t_R, &mHPH_t_R_inv);

    // P_priori * H'
    arm_mat_mult_f32(&mP_priori, &mH_trans, &mK_temp);

    // K = P_priori * H' * inv(H * P_priori * H' + R)
    arm_mat_mult_f32(&mK_temp, &mHPH_t_R_inv, &mK);

    // Calculate X hat posteriori /////////////////////////////////////
    //////////////////////////////////////////////////////////////////
    fZ[0][0] = z1;
    fZ[1][0] = z2;

    float32_t fZ_h[2][1];  // z - h
    float32_t fKZ_h[NUM_STATES][1]; // K * (z - h)

    arm_matrix_instance_f32 mZ_h;
    arm_matrix_instance_f32 mKZ_h;

    arm_mat_init_f32(&mZ_h, 2, 1, (float32_t *)fZ_h);
    arm_mat_init_f32(&mKZ_h, NUM_STATES, 1, (float32_t *)fKZ_h);

    // z - h
    for (int i = 0; i < 2; i++) {
        fZ_h[i][0] = fZ[i][0] - fH_nonlinear[i];
    }

    // K * (z - h)
    arm_mat_mult_f32(&mK, &mZ_h, &mKZ_h);

    // x_hat_posteriori = x_hat_priori + K * (z - h)
    arm_mat_add_f32(&mX_hat_priori, &mKZ_h, &mX_hat_posteriori);

    // Calculate P   /////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////
    float32_t fI[NUM_STATES][NUM_STATES];  // Identity matrix
    float32_t fIKH[NUM_STATES][NUM_STATES]; // (I - K * H)
    float32_t fIKH_t[NUM_STATES][NUM_STATES]; // Transpose of (I - K * H)
    float32_t fIKHP[NUM_STATES][NUM_STATES]; // (I - K * H) * P_priori
    float32_t fKRK[NUM_STATES][NUM_STATES]; // K * R * K'

    arm_matrix_instance_f32 mI;
    arm_matrix_instance_f32 mIKH;
    arm_matrix_instance_f32 mIKH_t;
    arm_matrix_instance_f32 mIKHP;
    arm_matrix_instance_f32 mKRK;

    arm_mat_init_f32(&mI, NUM_STATES, NUM_STATES, (float32_t *)fI);
    arm_mat_init_f32(&mIKH, NUM_STATES, NUM_STATES, (float32_t *)fIKH);
    arm_mat_init_f32(&mIKH_t, NUM_STATES, NUM_STATES, (float32_t *)fIKH_t);
    arm_mat_init_f32(&mIKHP, NUM_STATES, NUM_STATES, (float32_t *)fIKHP);
    arm_mat_init_f32(&mKRK, NUM_STATES, NUM_STATES, (float32_t *)fKRK);

    // Identity matrix
    for (int i = 0; i < NUM_STATES; i++) {
        for (int j = 0; j < NUM_STATES; j++) {
            fI[i][j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    // I - K * H
    arm_mat_mult_f32(&mK, &mH_jacobian, &mIKH);
    arm_mat_sub_f32(&mI, &mIKH, &mIKH);

    // (I - K * H)' (Transpose)
    arm_mat_trans_f32(&mIKH, &mIKH_t);

    // (I - K * H) * P_priori * (I - K * H)'
    arm_mat_mult_f32(&mIKH, &mP_priori, &mIKHP);
    arm_mat_mult_f32(&mIKHP, &mIKH_t, &mP_posteriori);

    // K * R * K'
    arm_mat_mult_f32(&mK, &mR, &mKRK);
    arm_mat_mult_f32(&mKRK, &mK, &mKRK);

    // Add K * R * K' to P_posteriori
    arm_mat_add_f32(&mP_posteriori, &mKRK, &mP_posteriori);

}

void vEstimationSystemComputeEstimate(float32_t u1, float32_t u2, float32_t z1, float32_t z2){
    static char cFirstIteration = 1;

    if(cFirstIteration){
        // Initialize all matrices used in the extended kalman filter implementation
        arm_mat_init_f32(&mZ, 2, 1, &fZ[0][0]);
        arm_mat_init_f32(&mU, 2, 1, &fU[0][0]);
        arm_mat_init_f32(&mX_hat_posteriori, NUM_STATES, 1, &(fStateVector[0]));
        arm_mat_init_f32(&mX_hat_priori, NUM_STATES, 1, &fX_hat_priori[0]);
        arm_mat_init_f32(&mP_posteriori, NUM_STATES, NUM_STATES, &fP_posteriori[0][0]);
        arm_mat_init_f32(&mP_priori, NUM_STATES, NUM_STATES, &fP_priori[0][0]);
        arm_mat_init_f32(&mQ, NUM_STATES, NUM_STATES, &fQ[0][0]);
        arm_mat_init_f32(&mR, 2, 2, &fR[0][0]);
        arm_mat_init_f32(&mK, NUM_STATES, 2, &fK[0][0]);
        arm_mat_init_f32(&mF_jacobian, NUM_STATES, NUM_STATES, &fF_jacobian[0][0]);
        arm_mat_init_f32(&mH_jacobian, 2, NUM_STATES, &fH_jacobian[0][0]);
        arm_mat_init_f32(&mH_nonlinear, 2, 1, &fH_nonlinear[0]);


        vEstimationSystemComputeDynamicModel(u1, u2);
        vEstimationSystemEkfPredict();
        cFirstIteration = 0;
        return;
    }
    
    vEstimationSystemComputeObservation();
    vEstimationSystemEkfEstimate(z1, z2);

    vEstimationSystemComputeDynamicModel(u1, u2);
    vEstimationSystemEkfPredict();
}