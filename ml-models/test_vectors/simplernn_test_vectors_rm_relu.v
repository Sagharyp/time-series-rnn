// Test vectors for SimpleRNN simulation
// Format: [timestep] [AccX] [AccY] [AccZ] [expected_output]

// Sample 0, True class: 3, Predicted: 3
// Time step 0
// AccX = 0.098000, AccY = 1.021000, AccZ = 0.142000
// Time step 1
// AccX = 0.097000, AccY = 1.018000, AccZ = 0.137000
// Time step 2
// AccX = 0.097000, AccY = 1.022000, AccZ = 0.144000
// Time step 3
// AccX = 0.097000, AccY = 1.022000, AccZ = 0.169000
// Time step 4
// AccX = 0.095000, AccY = 1.022000, AccZ = 0.160000
// Time step 5
// AccX = 0.096000, AccY = 1.031000, AccZ = 0.160000
// Time step 6
// AccX = 0.097000, AccY = 1.022000, AccZ = 0.153000
// Time step 7
// AccX = 0.093000, AccY = 1.015000, AccZ = 0.140000
// Time step 8
// AccX = 0.094000, AccY = 1.012000, AccZ = 0.141000
// Time step 9
// AccX = 0.096000, AccY = 1.016000, AccZ = 0.141000
// Expected output class: 3
// Model prediction: 3

// Sample 1, True class: 3, Predicted: 3
// Time step 0
// AccX = -0.093000, AccY = 1.053000, AccZ = 0.052000
// Time step 1
// AccX = -0.075000, AccY = 1.059000, AccZ = 0.057000
// Time step 2
// AccX = -0.080000, AccY = 1.048000, AccZ = 0.068000
// Time step 3
// AccX = -0.102000, AccY = 1.051000, AccZ = 0.084000
// Time step 4
// AccX = -0.117000, AccY = 1.060000, AccZ = 0.098000
// Time step 5
// AccX = -0.104000, AccY = 1.046000, AccZ = 0.086000
// Time step 6
// AccX = -0.084000, AccY = 1.029000, AccZ = 0.074000
// Time step 7
// AccX = -0.097000, AccY = 1.040000, AccZ = 0.089000
// Time step 8
// AccX = -0.103000, AccY = 1.042000, AccZ = 0.078000
// Time step 9
// AccX = -0.105000, AccY = 1.051000, AccZ = 0.071000
// Expected output class: 3
// Model prediction: 3

// Sample 2, True class: 2, Predicted: 1
// Time step 0
// AccX = -0.245000, AccY = 0.806000, AccZ = 1.102000
// Time step 1
// AccX = -0.251000, AccY = 0.911000, AccZ = 1.057000
// Time step 2
// AccX = -0.210000, AccY = 0.919000, AccZ = 0.873000
// Time step 3
// AccX = -0.188000, AccY = 0.942000, AccZ = 0.650000
// Time step 4
// AccX = -0.131000, AccY = 0.768000, AccZ = 0.315000
// Time step 5
// AccX = -0.072000, AccY = 0.439000, AccZ = 0.229000
// Time step 6
// AccX = -0.089000, AccY = 0.298000, AccZ = 0.372000
// Time step 7
// AccX = -0.124000, AccY = 0.357000, AccZ = 0.465000
// Time step 8
// AccX = -0.207000, AccY = 0.782000, AccZ = 0.751000
// Time step 9
// AccX = -0.251000, AccY = 1.039000, AccZ = 0.853000
// Expected output class: 2
// Model prediction: 1

// Sample 3, True class: 3, Predicted: 3
// Time step 0
// AccX = -0.031000, AccY = 1.037000, AccZ = 0.099000
// Time step 1
// AccX = -0.012000, AccY = 1.074000, AccZ = 0.088000
// Time step 2
// AccX = -0.056000, AccY = 1.077000, AccZ = 0.072000
// Time step 3
// AccX = -0.087000, AccY = 1.048000, AccZ = 0.084000
// Time step 4
// AccX = -0.073000, AccY = 1.054000, AccZ = 0.108000
// Time step 5
// AccX = -0.038000, AccY = 1.042000, AccZ = 0.102000
// Time step 6
// AccX = -0.016000, AccY = 1.021000, AccZ = 0.099000
// Time step 7
// AccX = -0.011000, AccY = 1.016000, AccZ = 0.100000
// Time step 8
// AccX = -0.016000, AccY = 1.056000, AccZ = 0.088000
// Time step 9
// AccX = -0.027000, AccY = 1.039000, AccZ = 0.068000
// Expected output class: 3
// Model prediction: 3

// Sample 4, True class: 2, Predicted: 2
// Time step 0
// AccX = -0.250000, AccY = 0.290000, AccZ = 0.895000
// Time step 1
// AccX = -0.256000, AccY = 0.275000, AccZ = 0.897000
// Time step 2
// AccX = -0.266000, AccY = 0.260000, AccZ = 0.895000
// Time step 3
// AccX = -0.273000, AccY = 0.256000, AccZ = 0.894000
// Time step 4
// AccX = -0.275000, AccY = 0.251000, AccZ = 0.893000
// Time step 5
// AccX = -0.271000, AccY = 0.262000, AccZ = 0.893000
// Time step 6
// AccX = -0.289000, AccY = 0.250000, AccZ = 0.912000
// Time step 7
// AccX = -0.310000, AccY = 0.233000, AccZ = 0.900000
// Time step 8
// AccX = -0.324000, AccY = 0.225000, AccZ = 0.891000
// Time step 9
// AccX = -0.331000, AccY = 0.230000, AccZ = 0.896000
// Expected output class: 2
// Model prediction: 2

// Sample 5, True class: 3, Predicted: 3
// Time step 0
// AccX = -0.060000, AccY = 1.011000, AccZ = 0.202000
// Time step 1
// AccX = -0.038000, AccY = 1.025000, AccZ = 0.191000
// Time step 2
// AccX = -0.054000, AccY = 1.028000, AccZ = 0.195000
// Time step 3
// AccX = -0.082000, AccY = 1.009000, AccZ = 0.200000
// Time step 4
// AccX = -0.092000, AccY = 1.006000, AccZ = 0.185000
// Time step 5
// AccX = -0.090000, AccY = 1.015000, AccZ = 0.181000
// Time step 6
// AccX = -0.068000, AccY = 1.029000, AccZ = 0.193000
// Time step 7
// AccX = -0.045000, AccY = 1.051000, AccZ = 0.168000
// Time step 8
// AccX = -0.073000, AccY = 1.028000, AccZ = 0.183000
// Time step 9
// AccX = -0.086000, AccY = 1.024000, AccZ = 0.229000
// Expected output class: 3
// Model prediction: 3

// Sample 6, True class: 3, Predicted: 3
// Time step 0
// AccX = -0.011000, AccY = 1.030000, AccZ = 0.182000
// Time step 1
// AccX = -0.015000, AccY = 1.042000, AccZ = 0.180000
// Time step 2
// AccX = -0.020000, AccY = 1.038000, AccZ = 0.178000
// Time step 3
// AccX = -0.019000, AccY = 1.035000, AccZ = 0.170000
// Time step 4
// AccX = -0.012000, AccY = 1.028000, AccZ = 0.181000
// Time step 5
// AccX = -0.004000, AccY = 1.011000, AccZ = 0.181000
// Time step 6
// AccX = 0.001000, AccY = 1.002000, AccZ = 0.180000
// Time step 7
// AccX = -0.004000, AccY = 0.996000, AccZ = 0.182000
// Time step 8
// AccX = -0.013000, AccY = 0.999000, AccZ = 0.206000
// Time step 9
// AccX = -0.030000, AccY = 1.004000, AccZ = 0.222000
// Expected output class: 3
// Model prediction: 3

// Sample 7, True class: 2, Predicted: 2
// Time step 0
// AccX = 0.091000, AccY = 0.601000, AccZ = 0.786000
// Time step 1
// AccX = 0.088000, AccY = 0.602000, AccZ = 0.788000
// Time step 2
// AccX = 0.083000, AccY = 0.602000, AccZ = 0.789000
// Time step 3
// AccX = 0.077000, AccY = 0.601000, AccZ = 0.790000
// Time step 4
// AccX = 0.074000, AccY = 0.601000, AccZ = 0.791000
// Time step 5
// AccX = 0.075000, AccY = 0.600000, AccZ = 0.790000
// Time step 6
// AccX = 0.077000, AccY = 0.596000, AccZ = 0.791000
// Time step 7
// AccX = 0.078000, AccY = 0.591000, AccZ = 0.791000
// Time step 8
// AccX = 0.079000, AccY = 0.590000, AccZ = 0.792000
// Time step 9
// AccX = 0.079000, AccY = 0.593000, AccZ = 0.793000
// Expected output class: 2
// Model prediction: 2

// Sample 8, True class: 2, Predicted: 2
// Time step 0
// AccX = -0.018000, AccY = 0.298000, AccZ = 0.939000
// Time step 1
// AccX = -0.019000, AccY = 0.293000, AccZ = 0.935000
// Time step 2
// AccX = -0.016000, AccY = 0.304000, AccZ = 0.945000
// Time step 3
// AccX = -0.007000, AccY = 0.318000, AccZ = 0.950000
// Time step 4
// AccX = 0.001000, AccY = 0.329000, AccZ = 0.948000
// Time step 5
// AccX = 0.005000, AccY = 0.354000, AccZ = 0.947000
// Time step 6
// AccX = 0.002000, AccY = 0.368000, AccZ = 0.931000
// Time step 7
// AccX = -0.011000, AccY = 0.371000, AccZ = 0.919000
// Time step 8
// AccX = -0.026000, AccY = 0.364000, AccZ = 0.918000
// Time step 9
// AccX = -0.036000, AccY = 0.343000, AccZ = 0.917000
// Expected output class: 2
// Model prediction: 2

// Sample 9, True class: 2, Predicted: 3
// Time step 0
// AccX = -0.022000, AccY = 1.032000, AccZ = 0.152000
// Time step 1
// AccX = -0.027000, AccY = 1.024000, AccZ = 0.151000
// Time step 2
// AccX = -0.013000, AccY = 1.023000, AccZ = 0.151000
// Time step 3
// AccX = -0.006000, AccY = 1.023000, AccZ = 0.157000
// Time step 4
// AccX = -0.021000, AccY = 1.022000, AccZ = 0.160000
// Time step 5
// AccX = -0.028000, AccY = 1.019000, AccZ = 0.155000
// Time step 6
// AccX = -0.016000, AccY = 1.019000, AccZ = 0.148000
// Time step 7
// AccX = -0.004000, AccY = 1.023000, AccZ = 0.150000
// Time step 8
// AccX = -0.008000, AccY = 1.026000, AccZ = 0.152000
// Time step 9
// AccX = -0.015000, AccY = 1.028000, AccZ = 0.153000
// Expected output class: 2
// Model prediction: 3

