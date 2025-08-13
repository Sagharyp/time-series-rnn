import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import SimpleRNN, LSTM, GRU, Dense, Input, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

# 1. Load Data (CSV Files)
#data = pd.read_csv('cow1.csv')  # Load data from a single CSV file

# Change the path of the files to your own
csv_files = ['cow1.csv', 'cow2.csv', 'cow3.csv',
             'cow4.csv', 'cow5.csv', 'cow6.csv']
data = [pd.read_csv(file) for file in csv_files]  # Read all CSV files into a list
data = pd.concat(data, ignore_index=True)  # Merge all dataframes

# 2. Remove Rows Containing NaN Values
data = data.dropna(subset=['label'])  # Remove rows where 'label' column has NaN values
print(data['label'].value_counts())  # Check class distribution

# 3. Filter Only Required Classes ('GRZ', 'MOV', 'RES', 'RUS')
valid_classes = ['GRZ', 'MOV', 'RES', 'RUS']  # Desired classes
data = data[data['label'].isin(valid_classes)]  # Select only rows containing these classes

"""# Save filtered data to a CSV file | To check if data has been loaded correctly
data.to_csv('filtered_cow_data.csv', index=False)"""

# 4. Separate Features (X) and Labels (y)
# AccX, AccY, AccZ are the input features (motion along X-axis), while 'label' represents cow behavior categories.
X = data[['AccX', 'AccY', 'AccZ']].values  # Extract AccX, AccY, AccZ features
y = data[['label']].values  # 'label' contains categorical behavior data

# 5. Convert Categorical Data to Numerical Using One-Hot Encoding
# Convert cow behavior categories into numerical values using One-Hot Encoding.
onehot_encoder = OneHotEncoder(sparse_output=False)  # sparse_output=False ensures non-sparse output
y = onehot_encoder.fit_transform(y)  # Apply One-Hot Encoding

"""# 6. Normalize Data Using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)  # Scale data to the range of 0-1"""

# 7. Create a Function for Windowing Time-Series Data
# This function segments time-series data into windows, allowing the model to learn from past sequences.
def create_sequences(data, target, window_size):
    sequences = []
    labels = []
    # Create input (X) and output (y) sequences based on the window size
    for i in range(len(data) - window_size):
        sequences.append(data[i:i + window_size])  # Input: past window-size data
        labels.append(target[i + window_size])  # Output: next step in the sequence
    return np.array(sequences), np.array(labels)

# 8. Set Window Size and Apply Windowing
window_size = 10
X_sequences, y_sequences = create_sequences(X, y, window_size)

# 9. Split Data into Training and Test Sets
# Split data into 80% training and 20% test sets.
X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, shuffle=True)  # Shuffle enabled (was False before)

# 10. Define Neural Network Model
model = Sequential([
    Input(shape=(window_size, X_train.shape[2])),  # Input shape: window size and 3 features per window
    #LSTM(50, activation='tanh'),
    #GRU(50, activation='tanh'),
    SimpleRNN(128, activation='tanh', return_sequences=True),  # First SimpleRNN layer with return_sequences=True
    BatchNormalization(),  # Apply Batch Normalization
    #Dropout(0.05),  # Apply Dropout

    SimpleRNN(128, activation='tanh'),  # Second SimpleRNN layer with 128 units and 'tanh' activation
    BatchNormalization(),  # Apply Batch Normalization
    #Dropout(0.05),  # Apply Dropout
    Dense(y_train.shape[1], activation='softmax')  # Softmax activation for multi-class classification
])

# 11. Compile Model
# Compile the model using Adam optimizer and categorical cross-entropy loss function.
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)  # Set optimizer and learning rate
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 12. Train the Model
#early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)  # Stop training if no improvement for 10 epochs
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
history = model.fit(X_train, y_train, epochs = 200, batch_size=128, validation_data=(X_test, y_test))
#DIKKAT 

# 13. Make Predictions Using the Model
y_pred = model.predict(X_test)  # Predict on test data
y_pred_classes = np.argmax(y_pred, axis=1)  # Get predicted class indices
y_true_classes = np.argmax(y_test, axis=1)  # Get true class indices

# 14. Generate Classification Report and Confusion Matrix
print("Classified Performance Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=onehot_encoder.categories_[0]))




# Save the model in .h5 format | This file can be used in hls4ml (weights and biases are embedded)
model.save("SimpleRNN_model.h5")




# Plot Confusion Matrix (Helps to visualize which classes are being misclassified)
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(14, 10))  # Create new figure
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=onehot_encoder.categories_[0], yticklabels=onehot_encoder.categories_[0])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')  # Save Confusion Matrix plot (For server use)
#plt.show()  # Show Confusion Matrix plot (For PC use)






# 15. Plot Model Performance
# Plot training and validation accuracy over epochs.
plt.figure(figsize=(14, 10))  # Create new figure
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()  # Add legend
plt.title('Model Performance - Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('accuracy_plot.png')  # Save accuracy plot (For server use)
#plt.show()  # Show accuracy plot (For PC use) // DIKKAT 



def generate_test_vectors():
    """Generate test vectors for Verilog simulation from your dataset"""
    # Get a small sample from your test data (e.g., first 10 samples)
    test_samples = X_test[:10]
    test_labels = y_test[:10]
    
    # Initialize hidden state to zeros (as in your Verilog testbench)
    hidden_state = np.zeros((128,))
    
    # Open file for writing test vectors
    with open('gru_test_vectors.v', 'w') as f:
        f.write("// Test vectors for GRU simulation\n")
        f.write("// Format: [timestep] [AccX] [AccY] [AccZ] [expected_output]\n\n")
        
        # Track GRU states for prediction
        gru_layer = model.get_layer(index=0)  # Adjust based on your model structure
        
        for i, sample in enumerate(test_samples):
            # For window-based prediction, we need a sequence
            # Reshape sample for GRU input
            input_sequence = sample.reshape(1, sample.shape[0], sample.shape[1])
            
            # Get GRU output
            gru_output = gru_layer(input_sequence)
            
            # Get predicted class
            dense_output = model.layers[-1](gru_output)
            predicted_class = np.argmax(dense_output.numpy(), axis=1)[0]
            true_class = np.argmax(test_labels[i])
            
            # Write test vector
            f.write(f"// Sample {i}, True class: {true_class}, Predicted: {predicted_class}\n")
            
            # Write the input sequence (all time steps)
            for t in range(input_sequence.shape[1]):
                # Get features at this time step
                acc_x = input_sequence[0, t, 0]
                acc_y = input_sequence[0, t, 1]
                acc_z = input_sequence[0, t, 2]
                
                # Convert to fixed point
                acc_x_fixed = float_to_fixed(acc_x)
                acc_y_fixed = float_to_fixed(acc_y)
                acc_z_fixed = float_to_fixed(acc_z)
                
                # Write as Verilog assignments
                f.write(f"// Time step {t}\n")
                f.write(f"x_in[0*DATA_WIDTH +: DATA_WIDTH] = 16'h{acc_x_fixed & 0xFFFF:04x}; // AccX = {acc_x}\n")
                f.write(f"x_in[1*DATA_WIDTH +: DATA_WIDTH] = 16'h{acc_y_fixed & 0xFFFF:04x}; // AccY = {acc_y}\n")
                f.write(f"x_in[2*DATA_WIDTH +: DATA_WIDTH] = 16'h{acc_z_fixed & 0xFFFF:04x}; // AccZ = {acc_z}\n\n")
            
            # Write expected output (can compare in simulation)
            f.write(f"// Expected output class: {true_class}\n\n")
        
    print("Test vectors generated in gru_test_vectors.v")

# Call the function to generate test vectors
generate_test_vectors()