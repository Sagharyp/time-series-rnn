import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras.layers import Layer 

def custom_matmul(a, b):
   
    # Get shapes
    a_shape = tf.shape(a)
    b_shape = tf.shape(b)
    
    # For your project, we expect:
    # a: (batch_size, input_dim) or (batch_size, hidden_dim)
    # b: (input_dim, hidden_dim) or (hidden_dim, hidden_dim)
    
    batch_size = a_shape[0]
    a_cols = a_shape[1]  # Should match b_rows
    b_rows = b_shape[0]
    b_cols = b_shape[1]
    
    # Simple validation (optional, can remove for even more simplicity)
    # tf.debugging.assert_equal(a_cols, b_rows, message="Matrix dimensions must match")
    
    # Method 1: Using tf.einsum (most readable and efficient)
    # 'ij,jk->ik' means: (i,j) @ (j,k) -> (i,k)
    result = tf.einsum('ij,jk->ik', a, b)
    
    return result


def custom_matmul_explicit(a, b):
    """
    Even more explicit version using nested loops concept
    (but still using TensorFlow operations for efficiency)
    
    This shows the actual matrix multiplication mathematics:
    C[i,k] = sum over j of A[i,j] * B[j,k]
    """
    
    a_shape = tf.shape(a)
    b_shape = tf.shape(b)
    
    batch_size = a_shape[0]
    a_cols = a_shape[1]
    b_cols = b_shape[1]
    
    # Expand dimensions for broadcasting
    # a: (batch, a_cols, 1)
    # b: (1, a_cols, b_cols)  
    a_expanded = tf.expand_dims(a, axis=2)  # (batch, a_cols, 1)
    b_expanded = tf.expand_dims(b, axis=0)  # (1, a_cols, b_cols)
    
    # Element-wise multiplication and sum over the middle dimension
    # This is equivalent to: C[i,k] = sum_j(A[i,j] * B[j,k])
    products = a_expanded * b_expanded  # (batch, a_cols, b_cols)
    result = tf.reduce_sum(products, axis=1)  # (batch, b_cols)
    
    return result


def custom_matmul_very_simple(a, b):
    """
    Ultra-simplified version for educational purposes
    Shows the core matrix multiplication concept
    """
    
    # Just use tf.einsum - it's the clearest way to express matrix multiplication
    # Einstein summation notation: 'ij,jk->ik'
    # - i: batch dimension  
    # - j: shared dimension (input features or hidden units)
    # - k: output dimension (hidden units)
    return tf.einsum('ij,jk->ik', a, b)
