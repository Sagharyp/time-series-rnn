import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer
import numpy as np
from custom_matmul import custom_matmul  # Import your custom matmul function

class SimpleRNNCell(Layer):
   
    
    def __init__(self, units, activation='tanh', use_bias=True, **kwargs):
        super().__init__(**kwargs)
        
        # Core parameters only
        self.units = units
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        
        # State and output size (simplified)
        self.state_size = self.units
        self.output_size = self.units
    
    def build(self, input_shape):
        """
        Build the layer weights.
        SIMPLIFIED: Using default initializers only
        """
        # Input to hidden weights (W_x in RNN equations)
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            name="kernel",
            initializer='glorot_uniform'  # Default initializer
        )
        
        # Hidden to hidden weights (W_h in RNN equations)  
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            name="recurrent_kernel",
            initializer='orthogonal'  # Default for recurrent weights
        )
        
        # Bias vector (optional)
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                name="bias",
                initializer='zeros'
            )
        else:
            self.bias = None
    
    def call(self, inputs, states):
        """
        Forward pass for one time step.
        SIMPLIFIED: Removed all dropout and mask handling
        
        RNN equation: h_t = activation(W_x * x_t + W_h * h_{t-1} + b)
        """
        # Get previous hidden state
        prev_output = states[0] if isinstance(states, list) else states
        
        # Compute input transformation: W_x * x_t using custom matmul
        h = custom_matmul(inputs, self.kernel)
        if self.bias is not None:
            h += self.bias
        
        # Add recurrent connection: W_h * h_{t-1} using custom matmul
        output = h + custom_matmul(prev_output, self.recurrent_kernel)
        
        # Apply activation function
        if self.activation is not None:
            output = self.activation(output)
        
        # Return output and new state (they're the same for SimpleRNN)
        return output, [output]
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """Initialize hidden state with zeros"""
        return [tf.zeros((batch_size, self.state_size), dtype=dtype)]


class SimpleRNN(Layer):
   
    
    def __init__(self, units, activation='tanh', use_bias=True, 
                 return_sequences=False, **kwargs):
        super().__init__(**kwargs)
        
        # Core parameters only
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.return_sequences = return_sequences
        
        # Create the RNN cell
        self.cell = SimpleRNNCell(
            units=units,
            activation=activation,
            use_bias=use_bias
        )
    
    def build(self, input_shape):
        """Build the layer"""
        self.cell.build(input_shape)
        super().build(input_shape)
    
    def call(self, inputs, initial_state=None, training=None):
        """
        Forward pass through the RNN.
        FIXED: Using tf.unstack to handle symbolic tensors properly
        """
        batch_size = tf.shape(inputs)[0]
        
        # Initialize state if not provided
        if initial_state is None:
            initial_state = self.cell.get_initial_state(
                batch_size=batch_size, 
                dtype=inputs.dtype
            )
        
        # Process sequence using tf.unstack instead of Python loop
        # tf.unstack splits the tensor along time axis into a list of tensors
        input_list = tf.unstack(inputs, axis=1)  # Split along time dimension
        
        states = initial_state
        outputs = []
        
        # Process each time step
        for current_input in input_list:
            # Process current time step
            output, states = self.cell(current_input, states)
            
            if self.return_sequences:
                outputs.append(output)
        
        # Return appropriate output format
        if self.return_sequences:
            # Stack all outputs: shape (batch, timesteps, units)
            return tf.stack(outputs, axis=1)
        else:
            # Return only last output: shape (batch, units)
            return output
    
    def get_config(self):
        """Get layer configuration for saving/loading"""
        config = {
            'units': self.units,
            'activation': keras.activations.serialize(keras.activations.get(self.activation)),
            'use_bias': self.use_bias,
            'return_sequences': self.return_sequences,
        }
        base_config = super().get_config()
        return {**base_config, **config}