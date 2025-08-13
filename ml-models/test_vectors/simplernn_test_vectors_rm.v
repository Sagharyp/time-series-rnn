// Test vectors for SimpleRNN simulation
// Format: [timestep] [AccX] [AccY] [AccZ] [expected_output]

// Sample 0, True class: 2, Predicted: 2
// Time step 0
// AccX = -0.365000, AccY = 0.733000, AccZ = 0.619000
// Time step 1
// AccX = -0.368000, AccY = 0.724000, AccZ = 0.625000
// Time step 2
// AccX = -0.366000, AccY = 0.719000, AccZ = 0.623000
// Time step 3
// AccX = -0.366000, AccY = 0.719000, AccZ = 0.623000
// Time step 4
// AccX = -0.375000, AccY = 0.720000, AccZ = 0.622000
// Time step 5
// AccX = -0.380000, AccY = 0.714000, AccZ = 0.620000
// Time step 6
// AccX = -0.382000, AccY = 0.712000, AccZ = 0.624000
// Time step 7
// AccX = -0.379000, AccY = 0.710000, AccZ = 0.629000
// Time step 8
// AccX = -0.371000, AccY = 0.710000, AccZ = 0.632000
// Time step 9
// AccX = -0.360000, AccY = 0.712000, AccZ = 0.628000
// Expected output class: 2
// Model prediction: 2

// Sample 1, True class: 2, Predicted: 2
// Time step 0
// AccX = -0.026000, AccY = 0.986000, AccZ = 0.268000
// Time step 1
// AccX = -0.027000, AccY = 0.985000, AccZ = 0.267000
// Time step 2
// AccX = -0.034000, AccY = 0.987000, AccZ = 0.267000
// Time step 3
// AccX = -0.034000, AccY = 0.988000, AccZ = 0.268000
// Time step 4
// AccX = -0.029000, AccY = 0.986000, AccZ = 0.269000
// Time step 5
// AccX = -0.026000, AccY = 0.984000, AccZ = 0.270000
// Time step 6
// AccX = -0.030000, AccY = 0.984000, AccZ = 0.272000
// Time step 7
// AccX = -0.037000, AccY = 0.984000, AccZ = 0.271000
// Time step 8
// AccX = -0.037000, AccY = 0.983000, AccZ = 0.270000
// Time step 9
// AccX = -0.030000, AccY = 0.983000, AccZ = 0.270000
// Expected output class: 2
// Model prediction: 2

// Sample 2, True class: 3, Predicted: 3
// Time step 0
// AccX = 0.053000, AccY = 0.877000, AccZ = 0.057000
// Time step 1
// AccX = -0.042000, AccY = 0.784000, AccZ = 0.109000
// Time step 2
// AccX = -0.098000, AccY = 0.786000, AccZ = 0.109000
// Time step 3
// AccX = -0.105000, AccY = 1.029000, AccZ = 0.113000
// Time step 4
// AccX = -0.107000, AccY = 1.405000, AccZ = 0.078000
// Time step 5
// AccX = -0.120000, AccY = 1.472000, AccZ = 0.095000
// Time step 6
// AccX = -0.068000, AccY = 1.118000, AccZ = 0.236000
// Time step 7
// AccX = 0.013000, AccY = 0.797000, AccZ = 0.279000
// Time step 8
// AccX = 0.101000, AccY = 0.770000, AccZ = 0.266000
// Time step 9
// AccX = 0.091000, AccY = 1.078000, AccZ = 0.164000
// Expected output class: 3
// Model prediction: 3

// Sample 3, True class: 2, Predicted: 2
// Time step 0
// AccX = -0.167000, AccY = 0.139000, AccZ = 0.950000
// Time step 1
// AccX = -0.172000, AccY = 0.134000, AccZ = 0.947000
// Time step 2
// AccX = -0.172000, AccY = 0.142000, AccZ = 0.947000
// Time step 3
// AccX = -0.170000, AccY = 0.156000, AccZ = 0.949000
// Time step 4
// AccX = -0.168000, AccY = 0.163000, AccZ = 0.949000
// Time step 5
// AccX = -0.165000, AccY = 0.159000, AccZ = 0.951000
// Time step 6
// AccX = -0.164000, AccY = 0.148000, AccZ = 0.950000
// Time step 7
// AccX = -0.164000, AccY = 0.140000, AccZ = 0.950000
// Time step 8
// AccX = -0.164000, AccY = 0.136000, AccZ = 0.949000
// Time step 9
// AccX = -0.165000, AccY = 0.137000, AccZ = 0.950000
// Expected output class: 2
// Model prediction: 2

// Sample 4, True class: 2, Predicted: 2
// Time step 0
// AccX = -0.281000, AccY = 0.366000, AccZ = 0.876000
// Time step 1
// AccX = -0.281000, AccY = 0.366000, AccZ = 0.874000
// Time step 2
// AccX = -0.281000, AccY = 0.366000, AccZ = 0.873000
// Time step 3
// AccX = -0.282000, AccY = 0.367000, AccZ = 0.874000
// Time step 4
// AccX = -0.281000, AccY = 0.366000, AccZ = 0.873000
// Time step 5
// AccX = -0.282000, AccY = 0.366000, AccZ = 0.876000
// Time step 6
// AccX = -0.282000, AccY = 0.366000, AccZ = 0.880000
// Time step 7
// AccX = -0.282000, AccY = 0.363000, AccZ = 0.881000
// Time step 8
// AccX = -0.283000, AccY = 0.359000, AccZ = 0.880000
// Time step 9
// AccX = -0.283000, AccY = 0.356000, AccZ = 0.878000
// Expected output class: 2
// Model prediction: 2

// Sample 5, True class: 3, Predicted: 3
// Time step 0
// AccX = -0.096000, AccY = 1.058000, AccZ = 0.068000
// Time step 1
// AccX = -0.100000, AccY = 1.074000, AccZ = 0.041000
// Time step 2
// AccX = -0.104000, AccY = 1.061000, AccZ = 0.011000
// Time step 3
// AccX = -0.118000, AccY = 1.066000, AccZ = 0.033000
// Time step 4
// AccX = -0.110000, AccY = 1.076000, AccZ = 0.052000
// Time step 5
// AccX = -0.084000, AccY = 1.053000, AccZ = 0.055000
// Time step 6
// AccX = -0.062000, AccY = 1.009000, AccZ = 0.112000
// Time step 7
// AccX = -0.064000, AccY = 0.997000, AccZ = 0.130000
// Time step 8
// AccX = -0.092000, AccY = 1.020000, AccZ = 0.084000
// Time step 9
// AccX = -0.098000, AccY = 1.036000, AccZ = 0.045000
// Expected output class: 3
// Model prediction: 3

// Sample 6, True class: 2, Predicted: 2
// Time step 0
// AccX = -0.297000, AccY = 0.288000, AccZ = 0.887000
// Time step 1
// AccX = -0.295000, AccY = 0.287000, AccZ = 0.885000
// Time step 2
// AccX = -0.293000, AccY = 0.290000, AccZ = 0.886000
// Time step 3
// AccX = -0.289000, AccY = 0.293000, AccZ = 0.886000
// Time step 4
// AccX = -0.287000, AccY = 0.293000, AccZ = 0.886000
// Time step 5
// AccX = -0.289000, AccY = 0.291000, AccZ = 0.891000
// Time step 6
// AccX = -0.293000, AccY = 0.290000, AccZ = 0.893000
// Time step 7
// AccX = -0.295000, AccY = 0.290000, AccZ = 0.894000
// Time step 8
// AccX = -0.293000, AccY = 0.290000, AccZ = 0.891000
// Time step 9
// AccX = -0.292000, AccY = 0.289000, AccZ = 0.889000
// Expected output class: 2
// Model prediction: 2

// Sample 7, True class: 3, Predicted: 3
// Time step 0
// AccX = -0.087000, AccY = 1.062000, AccZ = 0.066000
// Time step 1
// AccX = -0.084000, AccY = 1.079000, AccZ = 0.069000
// Time step 2
// AccX = -0.084000, AccY = 1.046000, AccZ = 0.091000
// Time step 3
// AccX = -0.057000, AccY = 1.030000, AccZ = 0.085000
// Time step 4
// AccX = -0.052000, AccY = 1.007000, AccZ = 0.077000
// Time step 5
// AccX = -0.055000, AccY = 1.022000, AccZ = 0.086000
// Time step 6
// AccX = -0.068000, AccY = 1.036000, AccZ = 0.070000
// Time step 7
// AccX = -0.068000, AccY = 1.031000, AccZ = 0.050000
// Time step 8
// AccX = -0.075000, AccY = 1.014000, AccZ = 0.062000
// Time step 9
// AccX = -0.081000, AccY = 1.032000, AccZ = 0.087000
// Expected output class: 3
// Model prediction: 3

// Sample 8, True class: 1, Predicted: 1
// Time step 0
// AccX = -0.072000, AccY = 0.844000, AccZ = 0.538000
// Time step 1
// AccX = -0.042000, AccY = 0.888000, AccZ = 0.576000
// Time step 2
// AccX = -0.005000, AccY = 0.870000, AccZ = 0.618000
// Time step 3
// AccX = 0.019000, AccY = 0.829000, AccZ = 0.627000
// Time step 4
// AccX = 0.026000, AccY = 0.790000, AccZ = 0.607000
// Time step 5
// AccX = 0.024000, AccY = 0.809000, AccZ = 0.595000
// Time step 6
// AccX = 0.013000, AccY = 0.852000, AccZ = 0.582000
// Time step 7
// AccX = -0.013000, AccY = 0.868000, AccZ = 0.546000
// Time step 8
// AccX = -0.039000, AccY = 0.877000, AccZ = 0.522000
// Time step 9
// AccX = -0.054000, AccY = 0.896000, AccZ = 0.511000
// Expected output class: 1
// Model prediction: 1

// Sample 9, True class: 2, Predicted: 2
// Time step 0
// AccX = -0.281000, AccY = 0.341000, AccZ = 0.883000
// Time step 1
// AccX = -0.277000, AccY = 0.343000, AccZ = 0.878000
// Time step 2
// AccX = -0.274000, AccY = 0.330000, AccZ = 0.874000
// Time step 3
// AccX = -0.277000, AccY = 0.321000, AccZ = 0.875000
// Time step 4
// AccX = -0.281000, AccY = 0.322000, AccZ = 0.874000
// Time step 5
// AccX = -0.286000, AccY = 0.332000, AccZ = 0.871000
// Time step 6
// AccX = -0.287000, AccY = 0.340000, AccZ = 0.867000
// Time step 7
// AccX = -0.287000, AccY = 0.341000, AccZ = 0.865000
// Time step 8
// AccX = -0.286000, AccY = 0.345000, AccZ = 0.872000
// Time step 9
// AccX = -0.283000, AccY = 0.352000, AccZ = 0.879000
// Expected output class: 2
// Model prediction: 2

