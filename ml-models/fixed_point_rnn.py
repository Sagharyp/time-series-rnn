import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np

class FixedPointConfig:
    """Configuration class for fixed-point parameters"""
    def __init__(self, integer_bits=8, fractional_bits=8):
        self.integer_bits = integer_bits
        self.fractional_bits = fractional_bits
        self.total_bits = integer_bits + fractional_bits
        self.scale_factor = 2 ** fractional_bits
        self.max_value = (2 ** (self.total_bits - 1) - 1) / self.scale_factor
        self.min_value = -(2 ** (self.total_bits - 1)) / self.scale_factor
        
    def __str__(self):
        return f"Q{self.integer_bits}.{self.fractional_bits} (total: {self.total_bits} bits)"

# Global configuration - you can change these values
WEIGHT_CONFIG = FixedPointConfig(integer_bits=4, fractional_bits=12)  # Q4.12 for weights
ACTIVATION_CONFIG = FixedPointConfig(integer_bits=8, fractional_bits=8)  # Q8.8 for activations
INPUT_CONFIG = FixedPointConfig(integer_bits=8, fractional_bits=8)  # Q8.8 for inputs

def float_to_fixed_point(value, config):
    """Convert floating point to fixed point representation (still using float)"""
    # Clip to valid range
    clipped = tf.clip_by_value(value, config.min_value, config.max_value)
    
    # Scale to integer representation
    scaled = clipped * config.scale_factor
    
    # Round to simulate integer quantization
    quantized_int = tf.round(scaled)
    
    # Convert back to float (simulating fixed-point)
    quantized_float = quantized_int / config.scale_factor
    
    return quantized_float

def quantize_tensor(tensor, config, name="tensor"):
    """Quantize a tensor with optional logging"""
    original_range = (tf.reduce_min(tensor), tf.reduce_max(tensor))
    quantized = float_to_fixed_point(tensor, config)
    quantized_range = (tf.reduce_min(quantized), tf.reduce_max(quantized))
    
    # Optional: Add logging to track quantization effects
    tf.print(f"{name} - Original range: {original_range}, Quantized range: {quantized_range}")
    
    return quantized

def custom_matmul_fixed_point(a, b, a_config=ACTIVATION_CONFIG, b_config=WEIGHT_CONFIG, output_config=ACTIVATION_CONFIG):
    """
    Matrix multiplication with fixed-point simulation
    
    Args:
        a: Input tensor (activations)
        b: Weight tensor 
        a_config: Fixed-point config for input
        b_config: Fixed-point config for weights
        output_config: Fixed-point config for output
    """
    
    # Quantize inputs (simulate ADC quantization)
    a_quantized = quantize_tensor(a, a_config, "input_activation")
    
    # Quantize weights (these would be pre-quantized in FPGA)
    b_quantized = quantize_tensor(b, b_config, "weights")
    
    # Perform matrix multiplication
    result = tf.einsum('ij,jk->ik', a_quantized, b_quantized)
    
    # The result of multiplication needs different precision handling
    # In fixed-point multiplication: Q(i1,f1) * Q(i2,f2) = Q(i1+i2, f1+f2)
    intermediate_config = FixedPointConfig(
        integer_bits=a_config.integer_bits + b_config.integer_bits,
        fractional_bits=a_config.fractional_bits + b_config.fractional_bits
    )
    
    # Quantize intermediate result
    result_intermediate = quantize_tensor(result, intermediate_config, "matmul_intermediate")
    
    # Scale back to desired output format
    result_final = quantize_tensor(result_intermediate, output_config, "matmul_output")
    
    return result_final

def fixed_point_activation(x, activation_fn, config=ACTIVATION_CONFIG):
    """Apply activation function with fixed-point quantization"""
    
    # Apply activation function first
    activated = activation_fn(x)
    
    # Quantize the result
    quantized = quantize_tensor(activated, config, "activation_output")
    
    return quantized

class SimpleRNNCell_FixedPoint(Layer):
    """Fixed-Point Simulation version of SimpleRNN Cell"""
    
    def __init__(self, units, activation='tanh', use_bias=True, 
                 weight_config=None, activation_config=None, **kwargs):
        super().__init__(**kwargs)
        
        self.units = units
        self.activation_fn = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        
        # Fixed-point configurations
        self.weight_config = weight_config or WEIGHT_CONFIG
        self.activation_config = activation_config or ACTIVATION_CONFIG
        
        self.state_size = self.units
        self.output_size = self.units
        
        print(f"SimpleRNNCell initialized with:")
        print(f"  - Weight precision: {self.weight_config}")
        print(f"  - Activation precision: {self.activation_config}")
    
    def build(self, input_shape):
        """Build layer weights with initial quantization"""
        
        # Input to hidden weights
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            name="kernel",
            initializer='glorot_uniform'
        )
        
        # Hidden to hidden weights
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            name="recurrent_kernel", 
            initializer='orthogonal'
        )
        
        # Bias
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                name="bias",
                initializer='zeros'
            )
        else:
            self.bias = None
    
    def call(self, inputs, states):
        """Forward pass with fixed-point simulation"""
        
        prev_output = states[0] if isinstance(states, list) else states
        
        # Quantize inputs (simulate sensor/ADC quantization)
        inputs_quantized = quantize_tensor(inputs, INPUT_CONFIG, "cell_input")
        prev_output_quantized = quantize_tensor(prev_output, self.activation_config, "prev_hidden")
        
        # Input transformation: W_x * x_t
        h = custom_matmul_fixed_point(
            inputs_quantized, 
            self.kernel,
            a_config=INPUT_CONFIG,
            b_config=self.weight_config,
            output_config=self.activation_config
        )
        
        # Add bias (quantized)
        if self.bias is not None:
            bias_quantized = quantize_tensor(self.bias, self.weight_config, "bias")
            h = h + bias_quantized
            h = quantize_tensor(h, self.activation_config, "after_bias")
        
        # Recurrent transformation: W_h * h_{t-1} 
        recurrent_output = custom_matmul_fixed_point(
            prev_output_quantized,
            self.recurrent_kernel,
            a_config=self.activation_config,
            b_config=self.weight_config, 
            output_config=self.activation_config
        )
        
        # Combine input and recurrent parts
        output = h + recurrent_output
        output = quantize_tensor(output, self.activation_config, "before_activation")
        
        # Apply activation with quantization
        if self.activation_fn is not None:
            output = fixed_point_activation(output, self.activation_fn, self.activation_config)
        
        return output, [output]
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """Initialize hidden state with zeros"""
        return [tf.zeros((batch_size, self.state_size), dtype=dtype)]

class SimpleRNN_FixedPoint(Layer):
    """Fixed-Point Simulation version of SimpleRNN"""
    
    def __init__(self, units, activation='tanh', use_bias=True, 
                 return_sequences=False, weight_config=None, activation_config=None, **kwargs):
        super().__init__(**kwargs)
        
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.return_sequences = return_sequences
        
        # Fixed-point configurations  
        self.weight_config = weight_config or WEIGHT_CONFIG
        self.activation_config = activation_config or ACTIVATION_CONFIG
        
        # Create the RNN cell with fixed-point simulation
        self.cell = SimpleRNNCell_FixedPoint(
            units=units,
            activation=activation,
            use_bias=use_bias,
            weight_config=self.weight_config,
            activation_config=self.activation_config
        )
    
    def build(self, input_shape):
        """Build the layer"""
        self.cell.build(input_shape)
        super().build(input_shape)
    
    def call(self, inputs, initial_state=None, training=None):
        """Forward pass with fixed-point simulation"""
        
        batch_size = tf.shape(inputs)[0]
        
        if initial_state is None:
            initial_state = self.cell.get_initial_state(
                batch_size=batch_size,
                dtype=inputs.dtype
            )
        
        # Process sequence
        input_list = tf.unstack(inputs, axis=1)
        
        states = initial_state
        outputs = []
        
        for current_input in input_list:
            output, states = self.cell(current_input, states)
            
            if self.return_sequences:
                outputs.append(output)
        
        if self.return_sequences:
            return tf.stack(outputs, axis=1)
        else:
            return output
    
    def get_config(self):
        """Get layer configuration"""
        config = {
            'units': self.units,
            'activation': tf.keras.activations.serialize(tf.keras.activations.get(self.activation)),
            'use_bias': self.use_bias,
            'return_sequences': self.return_sequences,
        }
        base_config = super().get_config()
        return {**base_config, **config}

class BatchNormalization_FixedPoint(Layer):
    """Fixed-Point Simulation version of Batch Normalization"""
    
    def __init__(self, momentum=0.99, epsilon=1e-3, center=True, scale=True, 
                 param_config=None, activation_config=None, **kwargs):
        super().__init__(**kwargs)
        
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.axis = -1
        
        # Fixed-point configurations
        self.param_config = param_config or WEIGHT_CONFIG  # For gamma, beta
        self.activation_config = activation_config or ACTIVATION_CONFIG  # For outputs
        
        print(f"BatchNorm initialized with:")
        print(f"  - Parameter precision: {self.param_config}")
        print(f"  - Activation precision: {self.activation_config}")
    
    def build(self, input_shape):
        """Build the layer weights"""
        feature_size = input_shape[self.axis]
        
        if self.scale:
            self.gamma = self.add_weight(
                shape=(feature_size,),
                name="gamma",
                initializer='ones',
                trainable=True
            )
        else:
            self.gamma = None
            
        if self.center:
            self.beta = self.add_weight(
                shape=(feature_size,),
                name="beta",
                initializer='zeros', 
                trainable=True
            )
        else:
            self.beta = None
        
        # Moving averages
        self.moving_mean = self.add_weight(
            shape=(feature_size,),
            name="moving_mean",
            initializer='zeros',
            trainable=False
        )
        
        self.moving_variance = self.add_weight(
            shape=(feature_size,),
            name="moving_variance",
            initializer='ones',
            trainable=False
        )
    
    def call(self, inputs, training=None):
        """Forward pass with fixed-point simulation"""
        
        if training is None:
            training = False
        
        # Quantize inputs
        inputs_quantized = quantize_tensor(inputs, self.activation_config, "batchnorm_input")
        
        input_shape = tf.shape(inputs_quantized)
        ndim = len(inputs_quantized.shape)
        reduction_axes = list(range(ndim - 1))
        
        if training and self.trainable:
            # Training mode: use batch statistics
            batch_mean = tf.reduce_mean(inputs_quantized, axis=reduction_axes)
            batch_variance = tf.reduce_mean(tf.square(inputs_quantized - batch_mean), axis=reduction_axes)
            
            # Quantize statistics
            batch_mean_quantized = quantize_tensor(batch_mean, self.activation_config, "batch_mean")
            batch_variance_quantized = quantize_tensor(batch_variance, self.activation_config, "batch_variance")
            
            # Update moving averages
            new_moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * batch_mean_quantized
            new_moving_variance = self.momentum * self.moving_variance + (1 - self.momentum) * batch_variance_quantized
            
            self.moving_mean.assign(new_moving_mean)
            self.moving_variance.assign(new_moving_variance)
            
            mean = batch_mean_quantized
            variance = batch_variance_quantized
        else:
            # Inference mode: use moving averages
            mean = quantize_tensor(self.moving_mean, self.activation_config, "moving_mean")
            variance = quantize_tensor(self.moving_variance, self.activation_config, "moving_variance")
        
        # Batch normalization: (x - mean) / sqrt(var + epsilon)
        epsilon_quantized = quantize_tensor(tf.constant(self.epsilon), self.activation_config, "epsilon")
        variance_plus_epsilon = variance + epsilon_quantized
        variance_plus_epsilon = quantize_tensor(variance_plus_epsilon, self.activation_config, "var_plus_eps")
        
        # Square root (this is tricky in fixed-point, often approximated in hardware)
        sqrt_variance = tf.sqrt(variance_plus_epsilon)
        sqrt_variance = quantize_tensor(sqrt_variance, self.activation_config, "sqrt_variance")
        
        # Normalization
        normalized = (inputs_quantized - mean) / sqrt_variance
        normalized = quantize_tensor(normalized, self.activation_config, "normalized")
        
        # Apply scale and shift
        if self.scale and self.gamma is not None:
            gamma_quantized = quantize_tensor(self.gamma, self.param_config, "gamma")
            normalized = normalized * gamma_quantized
            normalized = quantize_tensor(normalized, self.activation_config, "after_scale")
            
        if self.center and self.beta is not None:
            beta_quantized = quantize_tensor(self.beta, self.param_config, "beta")
            normalized = normalized + beta_quantized
            normalized = quantize_tensor(normalized, self.activation_config, "final_output")
        
        return normalized
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = {
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
        }
        base_config = super().get_config()
        return {**base_config, **config}

def extract_quantized_weights(model, save_path="quantized_weights.txt"):
    """
    Extract quantized weights from trained model for Verilog implementation
    """
    print("Extracting quantized weights for FPGA deployment...")
    
    with open(save_path, 'w') as f:
        f.write("// Quantized weights for FPGA implementation\n")
        f.write(f"// Weight precision: {WEIGHT_CONFIG}\n")
        f.write(f"// Activation precision: {ACTIVATION_CONFIG}\n\n")
        
        layer_count = 0
        for layer in model.layers:
            if hasattr(layer, 'get_weights') and layer.get_weights():
                weights = layer.get_weights()
                f.write(f"// Layer {layer_count}: {layer.__class__.__name__}\n")
                
                for i, weight_matrix in enumerate(weights):
                    # Quantize the weight matrix
                    if 'bias' in layer.weights[i].name:
                        quantized_weights = float_to_fixed_point(weight_matrix, WEIGHT_CONFIG)
                    else:
                        quantized_weights = float_to_fixed_point(weight_matrix, WEIGHT_CONFIG)
                    
                    # Convert to fixed-point integers for Verilog
                    fixed_point_ints = np.round(quantized_weights * WEIGHT_CONFIG.scale_factor).astype(int)
                    
                    f.write(f"// Weight matrix {i}: {weight_matrix.shape}\n")
                    f.write(f"// Weight name: {layer.weights[i].name}\n")
                    
                    # Write in Verilog array format
                    if len(weight_matrix.shape) == 1:  # Bias vector
                        f.write("wire signed [{}:0] bias_{}_{}[{}:0] = {{\n".format(
                            WEIGHT_CONFIG.total_bits-1, layer_count, i, weight_matrix.shape[0]-1))
                        for j, val in enumerate(fixed_point_ints):
                            f.write(f"    {WEIGHT_CONFIG.total_bits}'sh{val & ((1 << WEIGHT_CONFIG.total_bits) - 1):0{(WEIGHT_CONFIG.total_bits+3)//4}x}")
                            if j < len(fixed_point_ints) - 1:
                                f.write(",")
                            f.write("\n")
                        f.write("};\n\n")
                    
                    elif len(weight_matrix.shape) == 2:  # Weight matrix
                        f.write("wire signed [{}:0] weights_{}_{} [{}:0][{}:0] = {{\n".format(
                            WEIGHT_CONFIG.total_bits-1, layer_count, i, 
                            weight_matrix.shape[0]-1, weight_matrix.shape[1]-1))
                        for row in range(weight_matrix.shape[0]):
                            f.write("    {")
                            for col in range(weight_matrix.shape[1]):
                                val = fixed_point_ints[row, col]
                                f.write(f"{WEIGHT_CONFIG.total_bits}'sh{val & ((1 << WEIGHT_CONFIG.total_bits) - 1):0{(WEIGHT_CONFIG.total_bits+3)//4}x}")
                                if col < weight_matrix.shape[1] - 1:
                                    f.write(", ")
                            f.write("}")
                            if row < weight_matrix.shape[0] - 1:
                                f.write(",")
                            f.write("\n")
                        f.write("};\n\n")
                
                layer_count += 1
    
    print(f"Quantized weights saved to {save_path}")

# # Usage example:
# def create_fixed_point_model(window_size, num_features, num_classes):
#     """Create model with fixed-point simulation layers"""
    
#     print(f"Creating model with fixed-point precision:")
#     print(f"  - Weights: {WEIGHT_CONFIG}")
#     print(f"  - Activations: {ACTIVATION_CONFIG}")
#     print(f"  - Inputs: {INPUT_CONFIG}")
    
#     model = tf.keras.Sequential([
#         tf.keras.layers.Input(shape=(window_size, num_features)),
        
#         # Your custom fixed-point layers
#         SimpleRNN_FixedPoint(128, activation='tanh', return_sequences=True),
#         BatchNormalization_FixedPoint(),
        
#         SimpleRNN_FixedPoint(128, activation='tanh', return_sequences=False),
#         BatchNormalization_FixedPoint(),
        
#         tf.keras.layers.Dense(num_classes, activation='softmax')
#     ])
    
#     return model

# Configuration functions to change precision easily
def set_weight_precision(integer_bits, fractional_bits):
    """Set global weight precision"""
    global WEIGHT_CONFIG
    WEIGHT_CONFIG = FixedPointConfig(integer_bits, fractional_bits)
    print(f"Weight precision set to: {WEIGHT_CONFIG}")

def set_activation_precision(integer_bits, fractional_bits):
    """Set global activation precision"""
    global ACTIVATION_CONFIG
    ACTIVATION_CONFIG = FixedPointConfig(integer_bits, fractional_bits)
    print(f"Activation precision set to: {ACTIVATION_CONFIG}")

def set_input_precision(integer_bits, fractional_bits):
    """Set global input precision"""
    global INPUT_CONFIG
    INPUT_CONFIG = FixedPointConfig(integer_bits, fractional_bits)
    print(f"Input precision set to: {INPUT_CONFIG}")

print("Fixed-Point Simulation System loaded!")
print("Current configurations:")
print(f"  - Weights: {WEIGHT_CONFIG}")
print(f"  - Activations: {ACTIVATION_CONFIG}")
print(f"  - Inputs: {INPUT_CONFIG}")
print("\nYou can change precision using:")
print("  set_weight_precision(integer_bits, fractional_bits)")
print("  set_activation_precision(integer_bits, fractional_bits)")
print("  set_input_precision(integer_bits, fractional_bits)")