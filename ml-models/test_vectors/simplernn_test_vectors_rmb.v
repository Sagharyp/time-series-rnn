// Test vectors for SimpleRNN simulation
// Format: [timestep] [AccX] [AccY] [AccZ] [expected_output]

// Sample 0, True class: 2, Predicted: 2
// Time step 0
// AccX = -0.060000, AccY = 0.980000, AccZ = 0.294000
// Time step 1
// AccX = -0.065000, AccY = 0.981000, AccZ = 0.297000
// Time step 2
// AccX = -0.067000, AccY = 0.981000, AccZ = 0.298000
// Time step 3
// AccX = -0.065000, AccY = 0.979000, AccZ = 0.299000
// Time step 4
// AccX = -0.063000, AccY = 0.977000, AccZ = 0.297000
// Time step 5
// AccX = -0.060000, AccY = 0.978000, AccZ = 0.297000
// Time step 6
// AccX = -0.061000, AccY = 0.980000, AccZ = 0.293000
// Time step 7
// AccX = -0.063000, AccY = 0.980000, AccZ = 0.289000
// Time step 8
// AccX = -0.062000, AccY = 0.980000, AccZ = 0.291000
// Time step 9
// AccX = -0.063000, AccY = 0.979000, AccZ = 0.292000
// Expected output class: 2
// Model prediction: 2

// Sample 1, True class: 2, Predicted: 2
// Time step 0
// AccX = -0.114000, AccY = 0.827000, AccZ = 0.600000
// Time step 1
// AccX = -0.113000, AccY = 0.815000, AccZ = 0.598000
// Time step 2
// AccX = -0.112000, AccY = 0.809000, AccZ = 0.598000
// Time step 3
// AccX = -0.113000, AccY = 0.809000, AccZ = 0.603000
// Time step 4
// AccX = -0.114000, AccY = 0.807000, AccZ = 0.606000
// Time step 5
// AccX = -0.114000, AccY = 0.813000, AccZ = 0.602000
// Time step 6
// AccX = -0.114000, AccY = 0.830000, AccZ = 0.594000
// Time step 7
// AccX = -0.113000, AccY = 0.850000, AccZ = 0.571000
// Time step 8
// AccX = -0.110000, AccY = 0.864000, AccZ = 0.543000
// Time step 9
// AccX = -0.107000, AccY = 0.875000, AccZ = 0.528000
// Expected output class: 2
// Model prediction: 2

// Sample 2, True class: 2, Predicted: 2
// Time step 0
// AccX = 0.022000, AccY = 0.465000, AccZ = 0.860000
// Time step 1
// AccX = 0.022000, AccY = 0.471000, AccZ = 0.860000
// Time step 2
// AccX = 0.022000, AccY = 0.475000, AccZ = 0.862000
// Time step 3
// AccX = 0.023000, AccY = 0.480000, AccZ = 0.869000
// Time step 4
// AccX = 0.026000, AccY = 0.483000, AccZ = 0.873000
// Time step 5
// AccX = 0.026000, AccY = 0.485000, AccZ = 0.872000
// Time step 6
// AccX = 0.025000, AccY = 0.484000, AccZ = 0.867000
// Time step 7
// AccX = 0.023000, AccY = 0.483000, AccZ = 0.877000
// Time step 8
// AccX = 0.020000, AccY = 0.481000, AccZ = 0.884000
// Time step 9
// AccX = 0.017000, AccY = 0.478000, AccZ = 0.883000
// Expected output class: 2
// Model prediction: 2

// Sample 3, True class: 2, Predicted: 2
// Time step 0
// AccX = 0.135000, AccY = 0.391000, AccZ = 0.885000
// Time step 1
// AccX = 0.150000, AccY = 0.375000, AccZ = 0.883000
// Time step 2
// AccX = 0.156000, AccY = 0.372000, AccZ = 0.887000
// Time step 3
// AccX = 0.148000, AccY = 0.382000, AccZ = 0.881000
// Time step 4
// AccX = 0.139000, AccY = 0.405000, AccZ = 0.873000
// Time step 5
// AccX = 0.134000, AccY = 0.416000, AccZ = 0.869000
// Time step 6
// AccX = 0.136000, AccY = 0.405000, AccZ = 0.873000
// Time step 7
// AccX = 0.146000, AccY = 0.387000, AccZ = 0.883000
// Time step 8
// AccX = 0.167000, AccY = 0.381000, AccZ = 0.891000
// Time step 9
// AccX = 0.189000, AccY = 0.386000, AccZ = 0.896000
// Expected output class: 2
// Model prediction: 2

// Sample 4, True class: 0, Predicted: 0
// Time step 0
// AccX = 0.488000, AccY = 0.590000, AccZ = 0.768000
// Time step 1
// AccX = 0.464000, AccY = 0.721000, AccZ = 0.764000
// Time step 2
// AccX = 0.343000, AccY = 0.749000, AccZ = 0.735000
// Time step 3
// AccX = 0.208000, AccY = 0.688000, AccZ = 0.666000
// Time step 4
// AccX = 0.137000, AccY = 0.607000, AccZ = 0.516000
// Time step 5
// AccX = 0.135000, AccY = 0.609000, AccZ = 0.522000
// Time step 6
// AccX = 0.265000, AccY = 0.677000, AccZ = 0.591000
// Time step 7
// AccX = 0.423000, AccY = 0.709000, AccZ = 0.724000
// Time step 8
// AccX = 0.489000, AccY = 0.647000, AccZ = 0.763000
// Time step 9
// AccX = 0.491000, AccY = 0.542000, AccZ = 0.730000
// Expected output class: 0
// Model prediction: 0

// Sample 5, True class: 2, Predicted: 2
// Time step 0
// AccX = -0.041000, AccY = 1.047000, AccZ = 0.111000
// Time step 1
// AccX = -0.043000, AccY = 1.045000, AccZ = 0.110000
// Time step 2
// AccX = -0.044000, AccY = 1.046000, AccZ = 0.107000
// Time step 3
// AccX = -0.043000, AccY = 1.043000, AccZ = 0.105000
// Time step 4
// AccX = -0.040000, AccY = 1.037000, AccZ = 0.108000
// Time step 5
// AccX = -0.040000, AccY = 1.037000, AccZ = 0.113000
// Time step 6
// AccX = -0.042000, AccY = 1.043000, AccZ = 0.112000
// Time step 7
// AccX = -0.041000, AccY = 1.042000, AccZ = 0.110000
// Time step 8
// AccX = -0.041000, AccY = 1.037000, AccZ = 0.108000
// Time step 9
// AccX = -0.040000, AccY = 1.043000, AccZ = 0.107000
// Expected output class: 2
// Model prediction: 2

// Sample 6, True class: 2, Predicted: 2
// Time step 0
// AccX = -0.074000, AccY = 0.491000, AccZ = 0.845000
// Time step 1
// AccX = -0.080000, AccY = 0.494000, AccZ = 0.834000
// Time step 2
// AccX = -0.085000, AccY = 0.500000, AccZ = 0.824000
// Time step 3
// AccX = -0.087000, AccY = 0.510000, AccZ = 0.812000
// Time step 4
// AccX = -0.088000, AccY = 0.516000, AccZ = 0.812000
// Time step 5
// AccX = -0.088000, AccY = 0.516000, AccZ = 0.812000
// Time step 6
// AccX = -0.076000, AccY = 0.498000, AccZ = 0.844000
// Time step 7
// AccX = -0.076000, AccY = 0.498000, AccZ = 0.844000
// Time step 8
// AccX = -0.073000, AccY = 0.498000, AccZ = 0.853000
// Time step 9
// AccX = -0.074000, AccY = 0.500000, AccZ = 0.862000
// Expected output class: 2
// Model prediction: 2

// Sample 7, True class: 3, Predicted: 3
// Time step 0
// AccX = -0.035000, AccY = 1.022000, AccZ = 0.137000
// Time step 1
// AccX = -0.030000, AccY = 1.054000, AccZ = 0.134000
// Time step 2
// AccX = -0.008000, AccY = 1.059000, AccZ = 0.138000
// Time step 3
// AccX = 0.003000, AccY = 1.048000, AccZ = 0.139000
// Time step 4
// AccX = -0.002000, AccY = 1.036000, AccZ = 0.135000
// Time step 5
// AccX = -0.015000, AccY = 1.022000, AccZ = 0.134000
// Time step 6
// AccX = -0.020000, AccY = 1.019000, AccZ = 0.136000
// Time step 7
// AccX = -0.013000, AccY = 1.021000, AccZ = 0.135000
// Time step 8
// AccX = -0.011000, AccY = 1.014000, AccZ = 0.138000
// Time step 9
// AccX = -0.012000, AccY = 1.016000, AccZ = 0.138000
// Expected output class: 3
// Model prediction: 3

// Sample 8, True class: 3, Predicted: 3
// Time step 0
// AccX = -0.061000, AccY = 1.016000, AccZ = 0.177000
// Time step 1
// AccX = -0.130000, AccY = 1.002000, AccZ = 0.239000
// Time step 2
// AccX = -0.110000, AccY = 1.018000, AccZ = 0.190000
// Time step 3
// AccX = -0.003000, AccY = 1.048000, AccZ = 0.141000
// Time step 4
// AccX = 0.049000, AccY = 1.021000, AccZ = 0.152000
// Time step 5
// AccX = 0.013000, AccY = 1.020000, AccZ = 0.186000
// Time step 6
// AccX = -0.079000, AccY = 1.035000, AccZ = 0.209000
// Time step 7
// AccX = -0.113000, AccY = 1.033000, AccZ = 0.188000
// Time step 8
// AccX = -0.064000, AccY = 1.018000, AccZ = 0.188000
// Time step 9
// AccX = -0.021000, AccY = 1.004000, AccZ = 0.192000
// Expected output class: 3
// Model prediction: 3

// Sample 9, True class: 2, Predicted: 2
// Time step 0
// AccX = -0.022000, AccY = 0.873000, AccZ = 0.524000
// Time step 1
// AccX = -0.022000, AccY = 0.874000, AccZ = 0.525000
// Time step 2
// AccX = -0.020000, AccY = 0.871000, AccZ = 0.526000
// Time step 3
// AccX = -0.018000, AccY = 0.871000, AccZ = 0.527000
// Time step 4
// AccX = -0.016000, AccY = 0.875000, AccZ = 0.530000
// Time step 5
// AccX = -0.016000, AccY = 0.874000, AccZ = 0.529000
// Time step 6
// AccX = -0.015000, AccY = 0.872000, AccZ = 0.529000
// Time step 7
// AccX = -0.016000, AccY = 0.872000, AccZ = 0.528000
// Time step 8
// AccX = -0.019000, AccY = 0.870000, AccZ = 0.524000
// Time step 9
// AccX = -0.019000, AccY = 0.871000, AccZ = 0.523000
// Expected output class: 2
// Model prediction: 2

