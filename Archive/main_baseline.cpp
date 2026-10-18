#include <iostream>
#include "baseline_data.h"

int main() {
    std::cout << "--- gem5 EKF/UKF SLAM Baseline Execution ---" << std::endl;
    std::cout << "Loaded CNN Delta Z (m): " << TRAINED_KITTI_DELTA_Z << std::endl;

    float state_z = 0.0f;
    for (int i = 0; i < 10; ++i) {
        state_z += TRAINED_KITTI_DELTA_Z;
    }

    std::cout << "Final Estimated State Z: " << state_z << " m" << std::endl;
    return 0;
}