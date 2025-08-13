// Test vectors for SimpleRNN simulation
// Format: [timestep] [AccX] [AccY] [AccZ] [expected_output]

// Sample 0, True class: 3, Predicted: 3
// Time step 0
// AccX = -0.005000, AccY = 1.021000, AccZ = 0.142000
// Time step 1
// AccX = -0.026000, AccY = 1.018000, AccZ = 0.139000
// Time step 2
// AccX = -0.046000, AccY = 1.029000, AccZ = 0.140000
// Time step 3
// AccX = -0.059000, AccY = 1.037000, AccZ = 0.143000
// Time step 4
// AccX = -0.041000, AccY = 1.035000, AccZ = 0.139000
// Time step 5
// AccX = -0.022000, AccY = 1.039000, AccZ = 0.148000
// Time step 6
// AccX = 0.000000, AccY = 1.048000, AccZ = 0.135000
// Time step 7
// AccX = 0.009000, AccY = 1.027000, AccZ = 0.121000
// Time step 8
// AccX = -0.006000, AccY = 1.009000, AccZ = 0.122000
// Time step 9
// AccX = -0.031000, AccY = 1.011000, AccZ = 0.135000
// Expected output class: 3
// Model prediction: 3

// Sample 1, True class: 2, Predicted: 2
// Time step 0
// AccX = -0.163000, AccY = 0.712000, AccZ = 0.737000
// Time step 1
// AccX = -0.160000, AccY = 0.707000, AccZ = 0.736000
// Time step 2
// AccX = -0.160000, AccY = 0.708000, AccZ = 0.730000
// Time step 3
// AccX = -0.155000, AccY = 0.715000, AccZ = 0.716000
// Time step 4
// AccX = -0.153000, AccY = 0.732000, AccZ = 0.713000
// Time step 5
// AccX = -0.155000, AccY = 0.743000, AccZ = 0.720000
// Time step 6
// AccX = -0.162000, AccY = 0.725000, AccZ = 0.727000
// Time step 7
// AccX = -0.172000, AccY = 0.709000, AccZ = 0.732000
// Time step 8
// AccX = -0.176000, AccY = 0.693000, AccZ = 0.730000
// Time step 9
// AccX = -0.178000, AccY = 0.680000, AccZ = 0.722000
// Expected output class: 2
// Model prediction: 2

// Sample 2, True class: 2, Predicted: 2
// Time step 0
// AccX = 0.013000, AccY = 0.532000, AccZ = 0.832000
// Time step 1
// AccX = 0.013000, AccY = 0.533000, AccZ = 0.833000
// Time step 2
// AccX = 0.013000, AccY = 0.530000, AccZ = 0.834000
// Time step 3
// AccX = 0.014000, AccY = 0.526000, AccZ = 0.835000
// Time step 4
// AccX = 0.015000, AccY = 0.523000, AccZ = 0.838000
// Time step 5
// AccX = 0.016000, AccY = 0.523000, AccZ = 0.837000
// Time step 6
// AccX = 0.015000, AccY = 0.527000, AccZ = 0.837000
// Time step 7
// AccX = 0.015000, AccY = 0.532000, AccZ = 0.837000
// Time step 8
// AccX = 0.015000, AccY = 0.535000, AccZ = 0.837000
// Time step 9
// AccX = 0.015000, AccY = 0.538000, AccZ = 0.839000
// Expected output class: 2
// Model prediction: 2

// Sample 3, True class: 2, Predicted: 2
// Time step 0
// AccX = -0.234000, AccY = 0.431000, AccZ = 0.811000
// Time step 1
// AccX = -0.208000, AccY = 0.453000, AccZ = 0.842000
// Time step 2
// AccX = -0.163000, AccY = 0.467000, AccZ = 0.880000
// Time step 3
// AccX = -0.119000, AccY = 0.451000, AccZ = 0.929000
// Time step 4
// AccX = -0.130000, AccY = 0.433000, AccZ = 0.968000
// Time step 5
// AccX = -0.178000, AccY = 0.412000, AccZ = 0.956000
// Time step 6
// AccX = -0.229000, AccY = 0.384000, AccZ = 0.921000
// Time step 7
// AccX = -0.269000, AccY = 0.357000, AccZ = 0.880000
// Time step 8
// AccX = -0.281000, AccY = 0.330000, AccZ = 0.840000
// Time step 9
// AccX = -0.272000, AccY = 0.322000, AccZ = 0.834000
// Expected output class: 2
// Model prediction: 2

// Sample 4, True class: 2, Predicted: 2
// Time step 0
// AccX = -0.239000, AccY = 0.762000, AccZ = 0.642000
// Time step 1
// AccX = -0.236000, AccY = 0.765000, AccZ = 0.640000
// Time step 2
// AccX = -0.231000, AccY = 0.766000, AccZ = 0.637000
// Time step 3
// AccX = -0.230000, AccY = 0.768000, AccZ = 0.635000
// Time step 4
// AccX = -0.231000, AccY = 0.770000, AccZ = 0.633000
// Time step 5
// AccX = -0.232000, AccY = 0.768000, AccZ = 0.634000
// Time step 6
// AccX = -0.233000, AccY = 0.765000, AccZ = 0.635000
// Time step 7
// AccX = -0.233000, AccY = 0.763000, AccZ = 0.637000
// Time step 8
// AccX = -0.232000, AccY = 0.763000, AccZ = 0.639000
// Time step 9
// AccX = -0.232000, AccY = 0.765000, AccZ = 0.637000
// Expected output class: 2
// Model prediction: 2

// Sample 5, True class: 2, Predicted: 2
// Time step 0
// AccX = -0.196000, AccY = 0.406000, AccZ = 0.895000
// Time step 1
// AccX = -0.197000, AccY = 0.408000, AccZ = 0.896000
// Time step 2
// AccX = -0.196000, AccY = 0.407000, AccZ = 0.897000
// Time step 3
// AccX = -0.193000, AccY = 0.401000, AccZ = 0.893000
// Time step 4
// AccX = -0.198000, AccY = 0.400000, AccZ = 0.893000
// Time step 5
// AccX = -0.200000, AccY = 0.401000, AccZ = 0.890000
// Time step 6
// AccX = -0.191000, AccY = 0.400000, AccZ = 0.891000
// Time step 7
// AccX = -0.191000, AccY = 0.409000, AccZ = 0.896000
// Time step 8
// AccX = -0.192000, AccY = 0.415000, AccZ = 0.898000
// Time step 9
// AccX = -0.190000, AccY = 0.416000, AccZ = 0.895000
// Expected output class: 2
// Model prediction: 2

// Sample 6, True class: 2, Predicted: 2
// Time step 0
// AccX = -0.174000, AccY = 0.576000, AccZ = 0.792000
// Time step 1
// AccX = -0.172000, AccY = 0.564000, AccZ = 0.808000
// Time step 2
// AccX = -0.169000, AccY = 0.540000, AccZ = 0.818000
// Time step 3
// AccX = -0.167000, AccY = 0.513000, AccZ = 0.820000
// Time step 4
// AccX = -0.171000, AccY = 0.504000, AccZ = 0.828000
// Time step 5
// AccX = -0.177000, AccY = 0.512000, AccZ = 0.818000
// Time step 6
// AccX = -0.181000, AccY = 0.545000, AccZ = 0.801000
// Time step 7
// AccX = -0.180000, AccY = 0.597000, AccZ = 0.794000
// Time step 8
// AccX = -0.172000, AccY = 0.623000, AccZ = 0.785000
// Time step 9
// AccX = -0.148000, AccY = 0.623000, AccZ = 0.776000
// Expected output class: 2
// Model prediction: 2

// Sample 7, True class: 0, Predicted: 1
// Time step 0
// AccX = 0.257000, AccY = 0.906000, AccZ = 0.090000
// Time step 1
// AccX = 0.344000, AccY = 1.037000, AccZ = -0.008000
// Time step 2
// AccX = 0.412000, AccY = 0.946000, AccZ = -0.061000
// Time step 3
// AccX = 0.393000, AccY = 0.836000, AccZ = 0.081000
// Time step 4
// AccX = 0.422000, AccY = 1.033000, AccZ = 0.291000
// Time step 5
// AccX = 0.408000, AccY = 1.472000, AccZ = 0.144000
// Time step 6
// AccX = -0.011000, AccY = 1.633000, AccZ = 0.063000
// Time step 7
// AccX = -0.060000, AccY = 1.568000, AccZ = 0.372000
// Time step 8
// AccX = 0.249000, AccY = 1.262000, AccZ = 0.393000
// Time step 9
// AccX = 0.465000, AccY = 0.963000, AccZ = 0.287000
// Expected output class: 0
// Model prediction: 1

// Sample 8, True class: 3, Predicted: 3
// Time step 0
// AccX = -0.046000, AccY = 1.029000, AccZ = 0.166000
// Time step 1
// AccX = -0.043000, AccY = 1.032000, AccZ = 0.169000
// Time step 2
// AccX = -0.022000, AccY = 1.033000, AccZ = 0.158000
// Time step 3
// AccX = -0.017000, AccY = 1.034000, AccZ = 0.147000
// Time step 4
// AccX = -0.031000, AccY = 1.030000, AccZ = 0.154000
// Time step 5
// AccX = -0.032000, AccY = 1.009000, AccZ = 0.155000
// Time step 6
// AccX = -0.042000, AccY = 1.014000, AccZ = 0.159000
// Time step 7
// AccX = -0.047000, AccY = 1.016000, AccZ = 0.171000
// Time step 8
// AccX = -0.033000, AccY = 1.003000, AccZ = 0.185000
// Time step 9
// AccX = -0.026000, AccY = 0.996000, AccZ = 0.183000
// Expected output class: 3
// Model prediction: 3

// Sample 9, True class: 2, Predicted: 2
// Time step 0
// AccX = -0.092000, AccY = 1.000000, AccZ = 0.229000
// Time step 1
// AccX = -0.094000, AccY = 1.000000, AccZ = 0.229000
// Time step 2
// AccX = -0.093000, AccY = 0.993000, AccZ = 0.231000
// Time step 3
// AccX = -0.093000, AccY = 0.990000, AccZ = 0.231000
// Time step 4
// AccX = -0.093000, AccY = 0.992000, AccZ = 0.229000
// Time step 5
// AccX = -0.093000, AccY = 0.994000, AccZ = 0.231000
// Time step 6
// AccX = -0.093000, AccY = 0.995000, AccZ = 0.231000
// Time step 7
// AccX = -0.091000, AccY = 0.993000, AccZ = 0.230000
// Time step 8
// AccX = -0.090000, AccY = 0.992000, AccZ = 0.229000
// Time step 9
// AccX = -0.091000, AccY = 0.992000, AccZ = 0.229000
// Expected output class: 2
// Model prediction: 2

