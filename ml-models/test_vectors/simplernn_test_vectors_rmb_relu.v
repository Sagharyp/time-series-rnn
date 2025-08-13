// Test vectors for SimpleRNN simulation
// Format: [timestep] [AccX] [AccY] [AccZ] [expected_output]

// Sample 0, True class: 3, Predicted: 3
// Time step 0
// AccX = 0.049000, AccY = 1.036000, AccZ = 0.056000
// Time step 1
// AccX = 0.072000, AccY = 1.024000, AccZ = 0.065000
// Time step 2
// AccX = 0.081000, AccY = 1.056000, AccZ = 0.067000
// Time step 3
// AccX = 0.048000, AccY = 1.094000, AccZ = 0.052000
// Time step 4
// AccX = 0.015000, AccY = 1.108000, AccZ = 0.026000
// Time step 5
// AccX = -0.006000, AccY = 1.107000, AccZ = 0.016000
// Time step 6
// AccX = -0.037000, AccY = 1.103000, AccZ = 0.037000
// Time step 7
// AccX = -0.064000, AccY = 1.087000, AccZ = 0.052000
// Time step 8
// AccX = -0.023000, AccY = 1.060000, AccZ = 0.043000
// Time step 9
// AccX = 0.008000, AccY = 1.048000, AccZ = 0.058000
// Expected output class: 3
// Model prediction: 3

// Sample 1, True class: 2, Predicted: 2
// Time step 0
// AccX = -0.223000, AccY = 0.607000, AccZ = 0.809000
// Time step 1
// AccX = -0.223000, AccY = 0.591000, AccZ = 0.796000
// Time step 2
// AccX = -0.224000, AccY = 0.585000, AccZ = 0.785000
// Time step 3
// AccX = -0.230000, AccY = 0.583000, AccZ = 0.795000
// Time step 4
// AccX = -0.236000, AccY = 0.578000, AccZ = 0.810000
// Time step 5
// AccX = -0.241000, AccY = 0.577000, AccZ = 0.803000
// Time step 6
// AccX = -0.242000, AccY = 0.603000, AccZ = 0.797000
// Time step 7
// AccX = -0.232000, AccY = 0.609000, AccZ = 0.786000
// Time step 8
// AccX = -0.224000, AccY = 0.586000, AccZ = 0.776000
// Time step 9
// AccX = -0.222000, AccY = 0.560000, AccZ = 0.771000
// Expected output class: 2
// Model prediction: 2

// Sample 2, True class: 2, Predicted: 2
// Time step 0
// AccX = -0.249000, AccY = 0.221000, AccZ = 0.914000
// Time step 1
// AccX = -0.252000, AccY = 0.220000, AccZ = 0.912000
// Time step 2
// AccX = -0.253000, AccY = 0.220000, AccZ = 0.911000
// Time step 3
// AccX = -0.252000, AccY = 0.222000, AccZ = 0.911000
// Time step 4
// AccX = -0.248000, AccY = 0.226000, AccZ = 0.911000
// Time step 5
// AccX = -0.246000, AccY = 0.225000, AccZ = 0.910000
// Time step 6
// AccX = -0.248000, AccY = 0.223000, AccZ = 0.911000
// Time step 7
// AccX = -0.249000, AccY = 0.221000, AccZ = 0.909000
// Time step 8
// AccX = -0.249000, AccY = 0.221000, AccZ = 0.909000
// Time step 9
// AccX = -0.248000, AccY = 0.220000, AccZ = 0.912000
// Expected output class: 2
// Model prediction: 2

// Sample 3, True class: 3, Predicted: 3
// Time step 0
// AccX = -0.092000, AccY = 1.073000, AccZ = 0.054000
// Time step 1
// AccX = -0.102000, AccY = 1.054000, AccZ = 0.057000
// Time step 2
// AccX = -0.107000, AccY = 1.043000, AccZ = 0.089000
// Time step 3
// AccX = -0.091000, AccY = 1.023000, AccZ = 0.094000
// Time step 4
// AccX = -0.064000, AccY = 1.002000, AccZ = 0.087000
// Time step 5
// AccX = -0.083000, AccY = 1.025000, AccZ = 0.090000
// Time step 6
// AccX = -0.093000, AccY = 1.047000, AccZ = 0.066000
// Time step 7
// AccX = -0.074000, AccY = 1.052000, AccZ = 0.066000
// Time step 8
// AccX = -0.074000, AccY = 1.080000, AccZ = 0.061000
// Time step 9
// AccX = -0.092000, AccY = 1.083000, AccZ = 0.060000
// Expected output class: 3
// Model prediction: 3

// Sample 4, True class: 1, Predicted: 1
// Time step 0
// AccX = 0.039000, AccY = 0.742000, AccZ = 0.589000
// Time step 1
// AccX = 0.015000, AccY = 0.673000, AccZ = 0.595000
// Time step 2
// AccX = -0.037000, AccY = 0.745000, AccZ = 0.749000
// Time step 3
// AccX = -0.090000, AccY = 0.851000, AccZ = 0.812000
// Time step 4
// AccX = -0.113000, AccY = 0.846000, AccZ = 0.827000
// Time step 5
// AccX = -0.126000, AccY = 0.808000, AccZ = 0.818000
// Time step 6
// AccX = -0.141000, AccY = 0.792000, AccZ = 0.772000
// Time step 7
// AccX = -0.179000, AccY = 0.769000, AccZ = 0.713000
// Time step 8
// AccX = -0.189000, AccY = 0.732000, AccZ = 0.725000
// Time step 9
// AccX = -0.156000, AccY = 0.701000, AccZ = 0.764000
// Expected output class: 1
// Model prediction: 1

// Sample 5, True class: 3, Predicted: 3
// Time step 0
// AccX = -0.024000, AccY = 0.942000, AccZ = 0.141000
// Time step 1
// AccX = -0.009000, AccY = 0.976000, AccZ = 0.177000
// Time step 2
// AccX = 0.004000, AccY = 0.972000, AccZ = 0.189000
// Time step 3
// AccX = 0.035000, AccY = 1.041000, AccZ = 0.188000
// Time step 4
// AccX = 0.046000, AccY = 1.076000, AccZ = 0.114000
// Time step 5
// AccX = 0.029000, AccY = 1.034000, AccZ = 0.067000
// Time step 6
// AccX = 0.033000, AccY = 0.980000, AccZ = 0.037000
// Time step 7
// AccX = 0.055000, AccY = 0.917000, AccZ = 0.090000
// Time step 8
// AccX = 0.087000, AccY = 0.933000, AccZ = 0.115000
// Time step 9
// AccX = -0.010000, AccY = 1.214000, AccZ = 0.178000
// Expected output class: 3
// Model prediction: 3

// Sample 6, True class: 2, Predicted: 3
// Time step 0
// AccX = -0.035000, AccY = 1.021000, AccZ = 0.170000
// Time step 1
// AccX = -0.048000, AccY = 1.023000, AccZ = 0.185000
// Time step 2
// AccX = -0.057000, AccY = 1.019000, AccZ = 0.172000
// Time step 3
// AccX = -0.050000, AccY = 1.012000, AccZ = 0.166000
// Time step 4
// AccX = -0.048000, AccY = 1.013000, AccZ = 0.173000
// Time step 5
// AccX = -0.051000, AccY = 1.026000, AccZ = 0.177000
// Time step 6
// AccX = -0.037000, AccY = 1.015000, AccZ = 0.173000
// Time step 7
// AccX = -0.028000, AccY = 1.001000, AccZ = 0.173000
// Time step 8
// AccX = -0.044000, AccY = 1.015000, AccZ = 0.177000
// Time step 9
// AccX = -0.053000, AccY = 1.026000, AccZ = 0.180000
// Expected output class: 2
// Model prediction: 3

// Sample 7, True class: 2, Predicted: 2
// Time step 0
// AccX = -0.035000, AccY = 1.027000, AccZ = 0.250000
// Time step 1
// AccX = -0.057000, AccY = 1.029000, AccZ = 0.245000
// Time step 2
// AccX = -0.064000, AccY = 1.027000, AccZ = 0.241000
// Time step 3
// AccX = -0.063000, AccY = 1.035000, AccZ = 0.249000
// Time step 4
// AccX = -0.064000, AccY = 1.035000, AccZ = 0.242000
// Time step 5
// AccX = -0.065000, AccY = 1.006000, AccZ = 0.236000
// Time step 6
// AccX = -0.079000, AccY = 0.979000, AccZ = 0.232000
// Time step 7
// AccX = -0.089000, AccY = 0.970000, AccZ = 0.230000
// Time step 8
// AccX = -0.089000, AccY = 0.969000, AccZ = 0.234000
// Time step 9
// AccX = -0.090000, AccY = 0.970000, AccZ = 0.232000
// Expected output class: 2
// Model prediction: 2

// Sample 8, True class: 0, Predicted: 0
// Time step 0
// AccX = 0.463000, AccY = 0.558000, AccZ = 0.751000
// Time step 1
// AccX = 0.309000, AccY = 0.562000, AccZ = 0.760000
// Time step 2
// AccX = 0.108000, AccY = 0.551000, AccZ = 0.675000
// Time step 3
// AccX = 0.018000, AccY = 0.538000, AccZ = 0.629000
// Time step 4
// AccX = 0.040000, AccY = 0.515000, AccZ = 0.598000
// Time step 5
// AccX = 0.294000, AccY = 0.510000, AccZ = 0.661000
// Time step 6
// AccX = 0.380000, AccY = 0.678000, AccZ = 1.042000
// Time step 7
// AccX = 0.352000, AccY = 0.775000, AccZ = 1.049000
// Time step 8
// AccX = 0.190000, AccY = 0.741000, AccZ = 0.888000
// Time step 9
// AccX = 0.150000, AccY = 0.664000, AccZ = 0.814000
// Expected output class: 0
// Model prediction: 0

// Sample 9, True class: 0, Predicted: 0
// Time step 0
// AccX = 0.018000, AccY = 0.788000, AccZ = 0.842000
// Time step 1
// AccX = -0.040000, AccY = 0.704000, AccZ = 0.786000
// Time step 2
// AccX = -0.064000, AccY = 0.668000, AccZ = 0.733000
// Time step 3
// AccX = -0.064000, AccY = 0.643000, AccZ = 0.725000
// Time step 4
// AccX = -0.031000, AccY = 0.605000, AccZ = 0.723000
// Time step 5
// AccX = -0.016000, AccY = 0.581000, AccZ = 0.739000
// Time step 6
// AccX = 0.027000, AccY = 0.518000, AccZ = 0.737000
// Time step 7
// AccX = 0.172000, AccY = 0.519000, AccZ = 0.749000
// Time step 8
// AccX = 0.227000, AccY = 0.567000, AccZ = 0.724000
// Time step 9
// AccX = 0.082000, AccY = 0.578000, AccZ = 0.610000
// Expected output class: 0
// Model prediction: 0

