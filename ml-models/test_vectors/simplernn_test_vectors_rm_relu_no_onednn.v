// Test vectors for SimpleRNN simulation
// Format: [timestep] [AccX] [AccY] [AccZ] [expected_output]

// Sample 0, True class: 2, Predicted: 2
// Time step 0
// AccX = 0.039000, AccY = 0.546000, AccZ = 0.822000
// Time step 1
// AccX = 0.040000, AccY = 0.547000, AccZ = 0.822000
// Time step 2
// AccX = 0.040000, AccY = 0.545000, AccZ = 0.822000
// Time step 3
// AccX = 0.040000, AccY = 0.544000, AccZ = 0.822000
// Time step 4
// AccX = 0.040000, AccY = 0.546000, AccZ = 0.823000
// Time step 5
// AccX = 0.039000, AccY = 0.547000, AccZ = 0.823000
// Time step 6
// AccX = 0.040000, AccY = 0.546000, AccZ = 0.821000
// Time step 7
// AccX = 0.040000, AccY = 0.544000, AccZ = 0.821000
// Time step 8
// AccX = 0.040000, AccY = 0.542000, AccZ = 0.820000
// Time step 9
// AccX = 0.041000, AccY = 0.545000, AccZ = 0.819000
// Expected output class: 2
// Model prediction: 2

// Sample 1, True class: 1, Predicted: 1
// Time step 0
// AccX = -0.206000, AccY = 1.083000, AccZ = 0.067000
// Time step 1
// AccX = -0.040000, AccY = 1.112000, AccZ = 0.098000
// Time step 2
// AccX = 0.104000, AccY = 1.223000, AccZ = 0.275000
// Time step 3
// AccX = 0.209000, AccY = 1.276000, AccZ = 0.407000
// Time step 4
// AccX = 0.139000, AccY = 1.276000, AccZ = 0.401000
// Time step 5
// AccX = -0.028000, AccY = 1.128000, AccZ = 0.239000
// Time step 6
// AccX = -0.281000, AccY = 1.019000, AccZ = 0.150000
// Time step 7
// AccX = -0.380000, AccY = 1.029000, AccZ = 0.090000
// Time step 8
// AccX = -0.113000, AccY = 0.853000, AccZ = -0.108000
// Time step 9
// AccX = 0.079000, AccY = 0.817000, AccZ = -0.015000
// Expected output class: 1
// Model prediction: 1

// Sample 2, True class: 3, Predicted: 3
// Time step 0
// AccX = -0.015000, AccY = 1.052000, AccZ = 0.034000
// Time step 1
// AccX = -0.010000, AccY = 1.087000, AccZ = 0.016000
// Time step 2
// AccX = -0.059000, AccY = 1.082000, AccZ = 0.036000
// Time step 3
// AccX = -0.091000, AccY = 1.077000, AccZ = 0.077000
// Time step 4
// AccX = -0.052000, AccY = 1.060000, AccZ = 0.057000
// Time step 5
// AccX = -0.013000, AccY = 1.039000, AccZ = 0.044000
// Time step 6
// AccX = 0.001000, AccY = 1.032000, AccZ = 0.063000
// Time step 7
// AccX = 0.005000, AccY = 1.031000, AccZ = 0.082000
// Time step 8
// AccX = 0.007000, AccY = 1.041000, AccZ = 0.075000
// Time step 9
// AccX = 0.004000, AccY = 1.052000, AccZ = 0.037000
// Expected output class: 3
// Model prediction: 3

// Sample 3, True class: 1, Predicted: 2
// Time step 0
// AccX = -0.348000, AccY = 0.471000, AccZ = 0.838000
// Time step 1
// AccX = -0.311000, AccY = 0.480000, AccZ = 0.854000
// Time step 2
// AccX = -0.266000, AccY = 0.489000, AccZ = 0.865000
// Time step 3
// AccX = -0.231000, AccY = 0.487000, AccZ = 0.871000
// Time step 4
// AccX = -0.207000, AccY = 0.476000, AccZ = 0.865000
// Time step 5
// AccX = -0.207000, AccY = 0.453000, AccZ = 0.872000
// Time step 6
// AccX = -0.227000, AccY = 0.420000, AccZ = 0.871000
// Time step 7
// AccX = -0.269000, AccY = 0.407000, AccZ = 0.846000
// Time step 8
// AccX = -0.315000, AccY = 0.418000, AccZ = 0.839000
// Time step 9
// AccX = -0.337000, AccY = 0.417000, AccZ = 0.830000
// Expected output class: 1
// Model prediction: 2

// Sample 4, True class: 2, Predicted: 2
// Time step 0
// AccX = -0.247000, AccY = 0.222000, AccZ = 0.915000
// Time step 1
// AccX = -0.249000, AccY = 0.221000, AccZ = 0.914000
// Time step 2
// AccX = -0.252000, AccY = 0.220000, AccZ = 0.912000
// Time step 3
// AccX = -0.253000, AccY = 0.220000, AccZ = 0.911000
// Time step 4
// AccX = -0.252000, AccY = 0.222000, AccZ = 0.911000
// Time step 5
// AccX = -0.248000, AccY = 0.226000, AccZ = 0.911000
// Time step 6
// AccX = -0.246000, AccY = 0.225000, AccZ = 0.910000
// Time step 7
// AccX = -0.248000, AccY = 0.223000, AccZ = 0.911000
// Time step 8
// AccX = -0.249000, AccY = 0.221000, AccZ = 0.909000
// Time step 9
// AccX = -0.249000, AccY = 0.221000, AccZ = 0.909000
// Expected output class: 2
// Model prediction: 2

// Sample 5, True class: 2, Predicted: 2
// Time step 0
// AccX = -0.052000, AccY = 0.866000, AccZ = 0.535000
// Time step 1
// AccX = -0.049000, AccY = 0.866000, AccZ = 0.536000
// Time step 2
// AccX = -0.042000, AccY = 0.867000, AccZ = 0.537000
// Time step 3
// AccX = -0.036000, AccY = 0.866000, AccZ = 0.539000
// Time step 4
// AccX = -0.031000, AccY = 0.867000, AccZ = 0.540000
// Time step 5
// AccX = -0.030000, AccY = 0.870000, AccZ = 0.541000
// Time step 6
// AccX = -0.032000, AccY = 0.872000, AccZ = 0.541000
// Time step 7
// AccX = -0.035000, AccY = 0.872000, AccZ = 0.541000
// Time step 8
// AccX = -0.038000, AccY = 0.869000, AccZ = 0.540000
// Time step 9
// AccX = -0.043000, AccY = 0.866000, AccZ = 0.538000
// Expected output class: 2
// Model prediction: 2

// Sample 6, True class: 3, Predicted: 3
// Time step 0
// AccX = -0.044000, AccY = 1.037000, AccZ = 0.102000
// Time step 1
// AccX = -0.028000, AccY = 1.035000, AccZ = 0.090000
// Time step 2
// AccX = -0.010000, AccY = 1.034000, AccZ = 0.090000
// Time step 3
// AccX = 0.008000, AccY = 1.054000, AccZ = 0.059000
// Time step 4
// AccX = -0.015000, AccY = 1.064000, AccZ = 0.031000
// Time step 5
// AccX = -0.058000, AccY = 1.069000, AccZ = 0.039000
// Time step 6
// AccX = -0.091000, AccY = 1.091000, AccZ = 0.047000
// Time step 7
// AccX = -0.085000, AccY = 1.083000, AccZ = 0.032000
// Time step 8
// AccX = -0.053000, AccY = 1.054000, AccZ = 0.047000
// Time step 9
// AccX = -0.023000, AccY = 1.045000, AccZ = 0.088000
// Expected output class: 3
// Model prediction: 3

// Sample 7, True class: 3, Predicted: 3
// Time step 0
// AccX = -0.045000, AccY = 1.014000, AccZ = 0.202000
// Time step 1
// AccX = -0.044000, AccY = 1.013000, AccZ = 0.170000
// Time step 2
// AccX = -0.080000, AccY = 1.031000, AccZ = 0.156000
// Time step 3
// AccX = -0.093000, AccY = 1.047000, AccZ = 0.184000
// Time step 4
// AccX = -0.062000, AccY = 1.046000, AccZ = 0.209000
// Time step 5
// AccX = -0.031000, AccY = 1.018000, AccZ = 0.207000
// Time step 6
// AccX = -0.031000, AccY = 1.003000, AccZ = 0.181000
// Time step 7
// AccX = -0.045000, AccY = 0.991000, AccZ = 0.176000
// Time step 8
// AccX = -0.078000, AccY = 0.994000, AccZ = 0.201000
// Time step 9
// AccX = -0.065000, AccY = 1.003000, AccZ = 0.201000
// Expected output class: 3
// Model prediction: 3

// Sample 8, True class: 2, Predicted: 2
// Time step 0
// AccX = -0.159000, AccY = 0.417000, AccZ = 0.892000
// Time step 1
// AccX = -0.170000, AccY = 0.420000, AccZ = 0.893000
// Time step 2
// AccX = -0.173000, AccY = 0.419000, AccZ = 0.895000
// Time step 3
// AccX = -0.186000, AccY = 0.429000, AccZ = 0.896000
// Time step 4
// AccX = -0.173000, AccY = 0.419000, AccZ = 0.895000
// Time step 5
// AccX = -0.179000, AccY = 0.422000, AccZ = 0.894000
// Time step 6
// AccX = -0.187000, AccY = 0.428000, AccZ = 0.894000
// Time step 7
// AccX = -0.184000, AccY = 0.418000, AccZ = 0.883000
// Time step 8
// AccX = -0.192000, AccY = 0.434000, AccZ = 0.892000
// Time step 9
// AccX = -0.192000, AccY = 0.433000, AccZ = 0.894000
// Expected output class: 2
// Model prediction: 2

// Sample 9, True class: 2, Predicted: 2
// Time step 0
// AccX = -0.068000, AccY = 1.043000, AccZ = 0.084000
// Time step 1
// AccX = -0.075000, AccY = 1.047000, AccZ = 0.084000
// Time step 2
// AccX = -0.081000, AccY = 1.049000, AccZ = 0.085000
// Time step 3
// AccX = -0.084000, AccY = 1.043000, AccZ = 0.083000
// Time step 4
// AccX = -0.079000, AccY = 1.042000, AccZ = 0.083000
// Time step 5
// AccX = -0.073000, AccY = 1.046000, AccZ = 0.086000
// Time step 6
// AccX = -0.072000, AccY = 1.046000, AccZ = 0.084000
// Time step 7
// AccX = -0.077000, AccY = 1.047000, AccZ = 0.087000
// Time step 8
// AccX = -0.077000, AccY = 1.042000, AccZ = 0.088000
// Time step 9
// AccX = -0.074000, AccY = 1.043000, AccZ = 0.084000
// Expected output class: 2
// Model prediction: 2

