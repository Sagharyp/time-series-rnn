import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras.layers import Layer

class BatchNormalization(Layer):
    """
    Simplified BatchNormalization that works reliably with modern TensorFlow.
    
    ULTRA-SIMPLIFIED APPROACH:
    - Removed all complex update mechanisms
    - Uses simple conditional logic
    - Works in both training and inference
    - No deprecated TensorFlow functions
    """
    
    def __init__(self, momentum=0.99, epsilon=1e-3, center=True, scale=True, **kwargs):
        super().__init__(**kwargs)
        
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.axis = -1  # Always normalize last axis (features)
    
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
        
        # Moving averages (non-trainable)
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
        """
        ULTRA-SIMPLIFIED: Direct implementation without complex update mechanisms
        """
        # Handle training parameter
        if training is None:
            training = False
        
        # Calculate reduction axes (all except last)
        input_shape = tf.shape(inputs)
        ndim = len(inputs.shape)
        reduction_axes = list(range(ndim - 1))
        
        if training and self.trainable:
            # TRAINING: Use batch statistics
            batch_mean = tf.reduce_mean(inputs, axis=reduction_axes)
            batch_variance = tf.reduce_mean(tf.square(inputs - batch_mean), axis=reduction_axes)
            
            # Simple moving average updates (executed immediately)
            new_moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * batch_mean
            new_moving_variance = self.momentum * self.moving_variance + (1 - self.momentum) * batch_variance
            
            # Update the moving averages
            self.moving_mean.assign(new_moving_mean)
            self.moving_variance.assign(new_moving_variance)
            
            # Use batch statistics for normalization
            mean = batch_mean
            variance = batch_variance
        else:
            # INFERENCE: Use moving averages
            mean = self.moving_mean
            variance = self.moving_variance
        
        # Apply batch normalization
        normalized = (inputs - mean) / tf.sqrt(variance + self.epsilon)
        
        # Apply scale and shift
        if self.scale and self.gamma is not None:
            normalized = normalized * self.gamma
        if self.center and self.beta is not None:
            normalized = normalized + self.beta
            
        return normalized
    
    def compute_output_shape(self, input_shape):
        """Output shape is same as input shape"""
        return input_shape
    
    def get_config(self):
        """Get layer configuration"""
        config = {
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
        }
        base_config = super().get_config()
        return {**base_config, **config}
    
    def get_config(self):
        """Get layer configuration for saving/loading"""
        config = {
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
        }
        base_config = super().get_config()
        return {**base_config, **config}


class BatchNormalization_VerySimple(Layer):
    """
    Ultra-simplified version that works reliably
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.momentum = 0.99
        self.epsilon = 1e-3
    
    def build(self, input_shape):
        feature_size = input_shape[-1]
        
        self.gamma = self.add_weight(shape=(feature_size,), name="gamma", 
                                   initializer='ones', trainable=True)
        self.beta = self.add_weight(shape=(feature_size,), name="beta", 
                                  initializer='zeros', trainable=True)
        self.moving_mean = self.add_weight(shape=(feature_size,), name="moving_mean", 
                                         initializer='zeros', trainable=False)
        self.moving_variance = self.add_weight(shape=(feature_size,), name="moving_variance", 
                                             initializer='ones', trainable=False)
    
    def call(self, inputs, training=None):
        """
        Simplified call method that works with modern TensorFlow
        """
        if training is None:
            training = False
            
        # Calculate reduction axes
        ndim = len(inputs.shape)
        reduction_axes = list(range(ndim - 1))
        
        if training and self.trainable:
            # Use batch statistics
            mean = tf.reduce_mean(inputs, axis=reduction_axes)
            variance = tf.reduce_mean(tf.square(inputs - mean), axis=reduction_axes)
            
            # Update moving averages
            self.add_update(self.moving_mean.assign(
                self.momentum * self.moving_mean + (1 - self.momentum) * mean
            ))
            self.add_update(self.moving_variance.assign(
                self.momentum * self.moving_variance + (1 - self.momentum) * variance
            ))
        else:
            # Use moving averages
            mean = self.moving_mean
            variance = self.moving_variance
        
        # Apply batch normalization
        normalized = (inputs - mean) / tf.sqrt(variance + self.epsilon)
        return self.gamma * normalized + self.beta
    
    def compute_output_shape(self, input_shape):
        return input_shape




"""
SUMMARY OF ALL CHANGES MADE TO BATCHNORMALIZATION:

1. REMOVED COMPLEXITY:
   - All regularization (beta_regularizer, gamma_regularizer)
   - All constraints (beta_constraint, gamma_constraint) 
   - Custom initializers (now using 'ones' and 'zeros')
   - Synchronized batch norm for distributed training
   - Mask handling (not needed for your data)
   - Mixed precision casting (float16/bfloat16 handling)
   - Complex axis specification (always use last axis)
   - InputSpec validation
   - Error checking and edge case handling

2. SIMPLIFIED STRUCTURE:
   - Direct implementation of batch norm math
   - Clear separation of training vs inference mode
   - Straightforward parameter handling
   - Only essential parameters exposed

3. KEPT ESSENTIAL FEATURES:
   - Core batch normalization: (x - mean) / sqrt(var + eps)
   - Learnable scale (gamma) and shift (beta) parameters
   - Moving average tracking for inference
   - Training/inference mode switching
   - Proper weight initialization
   - Serialization support

4. YOUR PROJECT BENEFITS:
   - Much easier to understand the math
   - Clear training vs inference behavior
   - Easy to modify and experiment with
   - Perfect for educational purposes
   - Faster compilation (less overhead)

BATCH NORMALIZATION MATH:
Training:   output = gamma * (x - batch_mean) / sqrt(batch_var + epsilon) + beta
Inference:  output = gamma * (x - moving_mean) / sqrt(moving_var + epsilon) + beta

The moving averages are updated during training:
moving_mean = momentum * moving_mean + (1-momentum) * batch_mean
moving_var = momentum * moving_var + (1-momentum) * batch_var
"""