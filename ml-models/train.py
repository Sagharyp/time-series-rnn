import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout      
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

# Import your fixed-point simulation system
# Make sure the previous artifact is saved as 'fixed_point_rnn.py'
from fixed_point_rnn import (
    SimpleRNN_FixedPoint, 
    BatchNormalization_FixedPoint,
    extract_quantized_weights,
    set_weight_precision,
    set_activation_precision, 
    set_input_precision,
    WEIGHT_CONFIG,
    ACTIVATION_CONFIG,
    INPUT_CONFIG
)

# =============================================================================
# CONFIGURATION SECTION - MODIFY THESE FOR DIFFERENT PRECISION EXPERIMENTS
# =============================================================================

# Experiment with different fixed-point configurations
print("=== FIXED-POINT CONFIGURATION ===")

# Option 1: Conservative precision (more bits, higher accuracy)
# set_weight_precision(integer_bits=4, fractional_bits=12)    # Q4.12
# set_activation_precision(integer_bits=8, fractional_bits=8) # Q8.8
# set_input_precision(integer_bits=8, fractional_bits=8)      # Q8.8

# Option 2: Aggressive quantization (fewer bits, lower accuracy, smaller hardware)
# set_weight_precision(integer_bits=2, fractional_bits=6)     # Q2.6
# set_activation_precision(integer_bits=4, fractional_bits=4) # Q4.4  
# set_input_precision(integer_bits=4, fractional_bits=4)      # Q4.4

# Option 3: Balanced approach (default - good accuracy vs hardware trade-off)
set_weight_precision(integer_bits=4, fractional_bits=12)    # Q4.12 for weights
set_activation_precision(integer_bits=8, fractional_bits=8) # Q8.8 for activations
set_input_precision(integer_bits=8, fractional_bits=8)      # Q8.8 for inputs

print(f"Using precision configuration:")
print(f"  - Weights: {WEIGHT_CONFIG}")
print(f"  - Activations: {ACTIVATION_CONFIG}") 
print(f"  - Inputs: {INPUT_CONFIG}")
print("=" * 50)

# =============================================================================
# DATA LOADING AND PREPROCESSING (SAME AS ORIGINAL)
# =============================================================================

# 1. Load Data (CSV Files)
csv_files = ['data/cow1.csv', 'data/cow2.csv', 'data/cow3.csv',
             'data/cow4.csv', 'data/cow5.csv', 'data/cow6.csv']
             
try:
    data = [pd.read_csv(file) for file in csv_files]
    data = pd.concat(data, ignore_index=True)
    print(f"Loaded {len(data)} samples from {len(csv_files)} files")
except FileNotFoundError as e:
    print(f"Warning: Could not find data files. Using dummy data for demonstration.")
    print(f"Error: {e}")
    # Create dummy data for demonstration
    np.random.seed(42)
    n_samples = 10000
    data = pd.DataFrame({
        'AccX': np.random.randn(n_samples) * 2,
        'AccY': np.random.randn(n_samples) * 2, 
        'AccZ': np.random.randn(n_samples) * 2 + 9.8,  # Gravity bias
        'label': np.random.choice(['GRZ', 'MOV', 'RES', 'RUS'], n_samples)
    })

# 2. Remove Rows Containing NaN Values
data = data.dropna(subset=['label'])
print(f"Class distribution:")
print(data['label'].value_counts())

# 3. Filter Only Required Classes
valid_classes = ['GRZ', 'MOV', 'RES', 'RUS']
data = data[data['label'].isin(valid_classes)]
print(f"Filtered data: {len(data)} samples")

# 4. Separate Features (X) and Labels (y)
X = data[['AccX', 'AccY', 'AccZ']].values
y = data[['label']].values

# 5. Convert Categorical Data to Numerical Using One-Hot Encoding
onehot_encoder = OneHotEncoder(sparse_output=False)
y = onehot_encoder.fit_transform(y)
print(f"Classes: {onehot_encoder.categories_[0]}")

# Optional: Normalize data (you can experiment with/without this)
use_normalization = True
if use_normalization:
    print("Applying MinMaxScaler normalization...")
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
else:
    print("No normalization applied.")

# 7. Create Sequences for Time-Series Data
def create_sequences(data, target, window_size):
    sequences = []
    labels = []
    for i in range(len(data) - window_size):
        sequences.append(data[i:i + window_size])
        labels.append(target[i + window_size])
    return np.array(sequences), np.array(labels)

# 8. Set Window Size and Apply Windowing
window_size = 10
X_sequences, y_sequences = create_sequences(X, y, window_size)
print(f"Created {len(X_sequences)} sequences with window size {window_size}")

# 9. Split Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(
    X_sequences, y_sequences, test_size=0.2, shuffle=True, random_state=42
)
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# =============================================================================
# MODEL DEFINITION WITH FIXED-POINT SIMULATION
# =============================================================================

print("\n=== BUILDING FIXED-POINT MODEL ===")

# Define model with fixed-point simulation layers
model = Sequential([
    Input(shape=(window_size, X_train.shape[2])),
    
    # First RNN layer with fixed-point simulation
    SimpleRNN_FixedPoint(128, activation='tanh', return_sequences=True),
    BatchNormalization_FixedPoint(),
    
    # Second RNN layer with fixed-point simulation  
    SimpleRNN_FixedPoint(128, activation='tanh', return_sequences=False),
    BatchNormalization_FixedPoint(),
    
    # Final classification layer (you can also make this fixed-point)
    Dense(y_train.shape[1], activation='softmax')
])

# 11. Compile Model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

print(f"\nModel uses fixed-point simulation with:")
print(f"  - Weight precision: {WEIGHT_CONFIG}")  
print(f"  - Activation precision: {ACTIVATION_CONFIG}")
print(f"  - Input precision: {INPUT_CONFIG}")

# =============================================================================
# TRAINING
# =============================================================================

print("\n=== TRAINING FIXED-POINT MODEL ===")

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.000001)
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train, 
    epochs=50,  # Reduced for faster experimentation
    batch_size=128,
    validation_data=(X_test, y_test),
    callbacks=[reduce_lr, early_stopping],
    verbose=1
)

# =============================================================================
# EVALUATION AND ANALYSIS
# =============================================================================

print("\n=== EVALUATING FIXED-POINT MODEL ===")

# 13. Make Predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# 14. Generate Classification Report
print("Fixed-Point Model Performance Report:")
print("=" * 50)
print(classification_report(y_true_classes, y_pred_classes, target_names=onehot_encoder.categories_[0]))

# Calculate additional metrics
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# =============================================================================
# WEIGHT EXTRACTION FOR FPGA DEPLOYMENT
# =============================================================================

print("\n=== EXTRACTING QUANTIZED WEIGHTS FOR FPGA ===")

# Save the model
model_filename = f"SimpleRNN_FixedPoint_W{WEIGHT_CONFIG.integer_bits}_{WEIGHT_CONFIG.fractional_bits}_A{ACTIVATION_CONFIG.integer_bits}_{ACTIVATION_CONFIG.fractional_bits}.h5"
model.save(model_filename)
print(f"Model saved as: {model_filename}")

# Extract quantized weights for Verilog
weights_filename = f"quantized_weights_W{WEIGHT_CONFIG.integer_bits}_{WEIGHT_CONFIG.fractional_bits}_A{ACTIVATION_CONFIG.integer_bits}_{ACTIVATION_CONFIG.fractional_bits}.v"
extract_quantized_weights(model, weights_filename)

# =============================================================================
# VISUALIZATION AND ANALYSIS
# =============================================================================

print("\n=== GENERATING VISUALIZATIONS ===")

# Plot Confusion Matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=onehot_encoder.categories_[0], 
            yticklabels=onehot_encoder.categories_[0])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix - Fixed Point Model\nWeights: {WEIGHT_CONFIG}, Activations: {ACTIVATION_CONFIG}')
confusion_filename = f'confusion_matrix_fp_W{WEIGHT_CONFIG.integer_bits}_{WEIGHT_CONFIG.fractional_bits}.png'
plt.savefig(confusion_filename)
plt.show()
print(f"Confusion matrix saved as: {confusion_filename}")

# Plot Training History
plt.figure(figsize=(15, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title(f'Model Accuracy - Fixed Point\nWeights: {WEIGHT_CONFIG}, Activations: {ACTIVATION_CONFIG}')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# Loss plot  
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Model Loss - Fixed Point')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
accuracy_filename = f'training_history_fp_W{WEIGHT_CONFIG.integer_bits}_{WEIGHT_CONFIG.fractional_bits}.png'
plt.savefig(accuracy_filename)
plt.show()
print(f"Training history saved as: {accuracy_filename}")

# =============================================================================
# TEST VECTOR GENERATION FOR VERILOG SIMULATION
# =============================================================================

def generate_fixed_point_test_vectors():
    """Generate test vectors with fixed-point values for Verilog simulation"""
    print("\n=== GENERATING FIXED-POINT TEST VECTORS ===")
    
    # Get test samples
    test_samples = X_test[:10]  # First 10 samples
    test_labels = y_test[:10]
    
    # Helper function to convert float to fixed-point hex
    def float_to_fixed_hex(value, config):
        # Clip to valid range
        clipped = np.clip(value, config.min_value, config.max_value)
        # Scale and round to integer
        fixed_int = np.round(clipped * config.scale_factor).astype(int)
        # Handle negative numbers (two's complement)
        if fixed_int < 0:
            fixed_int = (1 << config.total_bits) + fixed_int
        return fixed_int & ((1 << config.total_bits) - 1)
    
    test_vectors_filename = f'simplernn_test_vectors_fp_W{WEIGHT_CONFIG.integer_bits}_{WEIGHT_CONFIG.fractional_bits}.v'
    
    with open(test_vectors_filename, 'w') as f:
        f.write("// Fixed-Point Test vectors for SimpleRNN Verilog simulation\n")
        f.write(f"// Weight precision: {WEIGHT_CONFIG}\n") 
        f.write(f"// Activation precision: {ACTIVATION_CONFIG}\n")
        f.write(f"// Input precision: {INPUT_CONFIG}\n")
        f.write("// Format: [sample][timestep] input_data expected_output\n\n")
        
        f.write("// Parameter definitions\n")
        f.write(f"parameter WEIGHT_WIDTH = {WEIGHT_CONFIG.total_bits};\n")
        f.write(f"parameter WEIGHT_FRAC = {WEIGHT_CONFIG.fractional_bits};\n")
        f.write(f"parameter ACTIVATION_WIDTH = {ACTIVATION_CONFIG.total_bits};\n")
        f.write(f"parameter ACTIVATION_FRAC = {ACTIVATION_CONFIG.fractional_bits};\n")
        f.write(f"parameter INPUT_WIDTH = {INPUT_CONFIG.total_bits};\n")
        f.write(f"parameter INPUT_FRAC = {INPUT_CONFIG.fractional_bits};\n\n")
        
        for sample_idx, (sample, label) in enumerate(zip(test_samples, test_labels)):
            # Get model prediction for comparison
            sample_reshaped = sample.reshape(1, *sample.shape)
            prediction = model.predict(sample_reshaped, verbose=0)
            predicted_class = np.argmax(prediction)
            true_class = np.argmax(label)
            confidence = np.max(prediction)
            
            f.write(f"// Sample {sample_idx}\n")
            f.write(f"// True class: {true_class} ({onehot_encoder.categories_[0][true_class]})\n")
            f.write(f"// Predicted class: {predicted_class} ({onehot_encoder.categories_[0][predicted_class]})\n")
            f.write(f"// Confidence: {confidence:.4f}\n")
            
            # Write test vector in Verilog format
            f.write(f"// Test vector {sample_idx}:\n")
            f.write("initial begin\n")
            
            # Initialize sequence
            for t, timestep in enumerate(sample):
                acc_x, acc_y, acc_z = timestep
                
                # Convert to fixed-point hex
                acc_x_hex = float_to_fixed_hex(acc_x, INPUT_CONFIG)
                acc_y_hex = float_to_fixed_hex(acc_y, INPUT_CONFIG)
                acc_z_hex = float_to_fixed_hex(acc_z, INPUT_CONFIG)
                
                f.write(f"    // Timestep {t}: AccX={acc_x:.6f}, AccY={acc_y:.6f}, AccZ={acc_z:.6f}\n")
                f.write(f"    #10 input_valid = 1'b1;\n")
                f.write(f"    acc_x_in = {INPUT_CONFIG.total_bits}'h{acc_x_hex:0{(INPUT_CONFIG.total_bits+3)//4}x};\n")
                f.write(f"    acc_y_in = {INPUT_CONFIG.total_bits}'h{acc_y_hex:0{(INPUT_CONFIG.total_bits+3)//4}x};\n")
                f.write(f"    acc_z_in = {INPUT_CONFIG.total_bits}'h{acc_z_hex:0{(INPUT_CONFIG.total_bits+3)//4}x};\n")
                f.write(f"    #10 input_valid = 1'b0;\n")
            
            # Expected output
            f.write(f"    // Expected output class: {true_class}\n")
            f.write(f"    // Wait for processing and check output\n")
            f.write(f"    #100; // Wait for RNN processing\n")
            f.write(f"    if (output_class == {true_class}) begin\n")
            f.write(f"        $display(\"PASS: Sample {sample_idx} correctly classified as %d\", output_class);\n")
            f.write(f"    end else begin\n")
            f.write(f"        $display(\"FAIL: Sample {sample_idx} classified as %d, expected %d\", output_class, {true_class});\n")
            f.write(f"    end\n")
            f.write("end\n\n")
        
        # Add testbench template
        f.write("// Testbench module template:\n")
        f.write("/*\n")
        f.write("module simplernn_tb;\n")
        f.write("    // Clock and reset\n")
        f.write("    reg clk, reset;\n")
        f.write("    \n")
        f.write("    // Input signals\n")  
        f.write("    reg input_valid;\n")
        f.write(f"    reg signed [INPUT_WIDTH-1:0] acc_x_in, acc_y_in, acc_z_in;\n")
        f.write("    \n")
        f.write("    // Output signals\n")
        f.write("    wire output_valid;\n")
        f.write("    wire [1:0] output_class; // 2 bits for 4 classes\n")
        f.write("    \n")
        f.write("    // Instantiate your SimpleRNN module\n")
        f.write("    simplernn_top uut (\n")
        f.write("        .clk(clk),\n")
        f.write("        .reset(reset),\n") 
        f.write("        .input_valid(input_valid),\n")
        f.write("        .acc_x_in(acc_x_in),\n")
        f.write("        .acc_y_in(acc_y_in),\n")
        f.write("        .acc_z_in(acc_z_in),\n")
        f.write("        .output_valid(output_valid),\n")
        f.write("        .output_class(output_class)\n")
        f.write("    );\n")
        f.write("    \n")
        f.write("    // Clock generation\n")
        f.write("    always #5 clk = ~clk;\n")
        f.write("    \n")
        f.write("    // Include the test vectors generated above here\n")
        f.write("    \n")
        f.write("endmodule\n")
        f.write("*/\n")
    
    print(f"Fixed-point test vectors saved to: {test_vectors_filename}")

# Generate test vectors
generate_fixed_point_test_vectors()

# =============================================================================
# PRECISION ANALYSIS AND RECOMMENDATIONS  
# =============================================================================

def analyze_precision_effects():
    """Analyze the effects of quantization on different parts of the model"""
    print("\n=== PRECISION ANALYSIS ===")
    
    # Get weight statistics
    print("Weight Statistics:")
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'get_weights') and layer.get_weights():
            weights = layer.get_weights()
            for j, weight_matrix in enumerate(weights):
                weight_name = layer.weights[j].name
                w_min, w_max = np.min(weight_matrix), np.max(weight_matrix)
                w_std = np.std(weight_matrix)
                
                print(f"  Layer {i} ({layer.__class__.__name__}) - {weight_name}:")
                print(f"    Range: [{w_min:.6f}, {w_max:.6f}]")
                print(f"    Std: {w_std:.6f}")
                
                # Check if weights fit in current precision
                if w_min < WEIGHT_CONFIG.min_value or w_max > WEIGHT_CONFIG.max_value:
                    print(f"    ⚠️  WARNING: Weights exceed {WEIGHT_CONFIG} range!")
                    print(f"    Current range: [{WEIGHT_CONFIG.min_value:.6f}, {WEIGHT_CONFIG.max_value:.6f}]")
                else:
                    print(f"    ✅ Weights fit in {WEIGHT_CONFIG}")
    
    # Recommendations
    print("\nRecommendations:")
    print("1. If accuracy is too low, try:")
    print("   - Increase fractional bits for weights/activations")
    print("   - Use higher precision for critical layers")
    print("   - Add scaling factors in Verilog implementation")
    
    print("2. If hardware cost is too high, try:")
    print("   - Reduce precision gradually and monitor accuracy")
    print("   - Use different precision for different layers")
    print("   - Consider block floating-point representation")
    
    print("3. Current configuration analysis:")
    total_weight_bits = WEIGHT_CONFIG.total_bits
    total_activation_bits = ACTIVATION_CONFIG.total_bits 
    print(f"   - Weight storage: {total_weight_bits} bits per weight")
    print(f"   - Activation processing: {total_activation_bits} bits per activation")
    print(f"   - Multiply-accumulate: {total_weight_bits + total_activation_bits} bits intermediate")

analyze_precision_effects()

# =============================================================================
# COMPARISON WITH FLOATING-POINT MODEL (OPTIONAL)
# =============================================================================

def compare_with_floating_point():
    """Compare fixed-point model with standard floating-point model"""
    print("\n=== COMPARISON WITH FLOATING-POINT MODEL ===")
    
    # Create standard floating-point model for comparison
    fp_model = Sequential([
        Input(shape=(window_size, X_train.shape[2])),
        tf.keras.layers.SimpleRNN(128, activation='tanh', return_sequences=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.SimpleRNN(128, activation='tanh', return_sequences=False),
        tf.keras.layers.BatchNormalization(),
        Dense(y_train.shape[1], activation='softmax')
    ])
    
    fp_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), 
                     loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("Training floating-point model for comparison...")
    fp_history = fp_model.fit(X_train, y_train, epochs=20, batch_size=128, 
                              validation_data=(X_test, y_test), verbose=0)
    
    # Compare results
    fp_pred = fp_model.predict(X_test, verbose=0)
    fp_pred_classes = np.argmax(fp_pred, axis=1)
    
    fp_accuracy = np.mean(fp_pred_classes == y_true_classes)
    fixed_accuracy = np.mean(y_pred_classes == y_true_classes)
    
    print(f"Floating-point accuracy: {fp_accuracy:.4f}")
    print(f"Fixed-point accuracy:    {fixed_accuracy:.4f}")
    print(f"Accuracy difference:     {fp_accuracy - fixed_accuracy:.4f}")
    
    if abs(fp_accuracy - fixed_accuracy) < 0.05:
        print("✅ Fixed-point performance is close to floating-point!")
    elif fixed_accuracy < fp_accuracy - 0.05:
        print("⚠️  Fixed-point accuracy is significantly lower. Consider:")
        print("   - Increasing precision")
        print("   - Fine-tuning quantization parameters")
    else:
        print("🎉 Fixed-point performance matches or exceeds floating-point!")

# Uncomment to run comparison (takes additional time)
# compare_with_floating_point()

print("\n" + "=" * 70)
print("FIXED-POINT MODEL TRAINING COMPLETE!")
print("=" * 70)
print(f"Final test accuracy: {test_accuracy:.4f}")
print(f"Model saved as: {model_filename}")
print(f"Quantized weights: {weights_filename}")
print(f"Test vectors: simplernn_test_vectors_fp_W{WEIGHT_CONFIG.integer_bits}_{WEIGHT_CONFIG.fractional_bits}.v")
print("\nConfiguration used:")
print(f"  - Weights: {WEIGHT_CONFIG}")
print(f"  - Activations: {ACTIVATION_CONFIG}")
print(f"  - Inputs: {INPUT_CONFIG}")
print("=" * 70)

# =============================================================================
# EXPERIMENT SUGGESTIONS
# =============================================================================

print("\n=== SUGGESTED EXPERIMENTS ===")
print("Try different precision configurations by modifying the configuration section:")
print()
print("1. High precision (better accuracy, more hardware):")
print("   set_weight_precision(6, 10)     # Q6.10")  
print("   set_activation_precision(8, 8)  # Q8.8")
print()
print("2. Low precision (lower accuracy, less hardware):")
print("   set_weight_precision(2, 6)      # Q2.6") 
print("   set_activation_precision(4, 4)  # Q4.4")
print()
print("3. Mixed precision (balance accuracy vs hardware):")
print("   # Higher precision for weights, lower for activations")
print("   set_weight_precision(4, 12)     # Q4.12")
print("   set_activation_precision(6, 6)  # Q6.6") 
print()
print("4. Monitor the accuracy vs hardware trade-off and choose optimal configuration!")
print("=" * 70)
