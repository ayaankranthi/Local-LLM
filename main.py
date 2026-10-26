import numpy as np

class Embeddinglayer:
    def __init__(self, vocab_size, embedding_dim):
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.02
        self.inputs = None
        
        self.grad_embeddings = np.zeros(self.embeddings.shape)
        
    def forward(self, inputs):
        self.inputs = inputs
        return self.embeddings[inputs]
    
    def backward(self, prev_grad):
        grad_embeddings = np.zeros(self.embeddings.shape)
        np.add.at(grad_embeddings, self.inputs, prev_grad)
        
        self.embeddings + grad_embeddings
        return grad_embeddings
    def get_params(self):
        pass
    
class Softmax:
    def __init__(self, x):
        self.softmax_out = None
        
    def forward(self, inputs):
        max_logit = np.max(inputs, axis=-1, keepdims=True)
        
        softmax_out = np.exp(inputs - max_logit) / np.sum(np.exp(inputs - max_logit), axis=-1, keepdims=True)
        self.softmax_out = softmax_out
        return softmax_out
    
    def backward(self, prev_gead):
      return prev_gead * self.softmax_out - self.softmax_out * np.sum(self.softmax_out * prev_gead, axis=-1, keepdims=True)  
    
class CrossEntropyLossWitSoftmax:
    def __init__(self):
        self.lse = None
        self.logits_flat = None
        self.lables_flat = None
        
    def forward(self, logits, lables):
        assert len(logits.shape) == 3
        B, T, V = logits.shape
        
        logits_flat = np.reshape(logits, (B * T, V))
        lables_flat = np.reshape(lables, (B * T))
        
        max_float = np.max(logits_flat, axis=-1, keepdims=True)
        diff = logits_flat - max_float
        
        lse = max_float + np.log(np.sum(np.exp(diff), axis=-1, keepdims=True))
        log_softmax = np.expand_dims(logits_flat[np.arange(B * T), lables_flat], axis=-1) - lse
        
        self.lse = lse
        self.logits_flat = logits_flat
        self.lables_flat = lables_flat
        self.B = B
        self.T = T
        self.V = V
        
        return -np.sum(log_softmax) / logits_flat.shape[0]
    
    def backward(self):
        pass
    
    def get_name(self):
        pass
    
class Linear:
    def __init__(self, dim_in, dim_out, include_bias=True):
        self.weights = np.random.randn(dim_in, dim_out) * np.sqrt(2.0 / (dim_in + dim_out))
        self.include_bias = include_bias
        
        if self.include_bias:
            self.bias = np.zeros((1, dim_out))
        else:
            self.bias = None
            
        self.inputs = None
        self.grad_weights = None
        self.grad_bias = None

    def forward(self, inputs):
        self.inputs = inputs 
        out = np.dot(inputs, self.weights)
        
        if self.include_bias:
            out += self.bias
            
        return out

    def backward(self, prev_grad):
        B, T, D_out = prev_grad.shape
        inputs_flat = self.inputs.reshape(-1, self.weights.shape[0])
        grad_flat = prev_grad.reshape(-1, D_out)
        
        self.grad_weights = np.dot(inputs_flat.T, grad_flat)
        
        if self.include_bias:
            self.grad_bias = np.sum(grad_flat, axis=0, keepdims=True)
            
        grad_inputs = np.dot(prev_grad, self.weights.T)
        
        return grad_inputs

    def get_params(self):
        params = {'W': self.weights}
        if self.include_bias:
            params['b'] = self.bias
        return params

class MultiHeadAttention:
    def __init__(self, dim_in, dim_out, n_heads):
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_heads = n_heads
        
        self.query_linear = Linear(dim_in, dim_out, incude_bais = False)
        self.key_linear = Linear(dim_in, dim_out, incude_bais = False)
        self.value_linear = Linear(dim_in, dim_out, incude_bais = False)
        self.softmax = Softmax()
        self.final_projection = Linear(dim_in, dim_out, include_bias=False)
        
        self.queries_matrix = None
        self.keys_matrix = None
        self.values_matrix = None
        self.attention_weights = None
        
    def forward(self, inputs, mask = None):
        B, T, D = inputs.shape
        
        queries = self.query_linear.forward(inputs)
        keys = self.key_linear.forward(inputs)
        values = self.value_linear.forward(inputs)
        
        queries = np.reshape(queries, (B, T, self.n_heads, self.dim_out // self.n_heads))
        keys = np.reshape(keys, (B, T, self.n_heads, self.dim_out // self.n_heads))
        values = np.reshape(values, (B, T, self.n_heads, self.dim_out // self.n_heads))
        
        queries = np.transpose(queries, axes=[0, 2, 1, 3])
        keys = np.transpose(keys, axes=[0, 2, 1, 3])
        values = np.transpose(values, axes=[0, 2, 1, 3])
        
        attention_scores = queries @ np.transpose(keys, axes=[0, 1, 3, 2])
        attention_scores = attention_scores / np.sqrt(keys.shape[-1])
        
        if mask is not None:
            attention_scores += mask
            
        attention_weights = self.softmax.forward(attention_scores)
        
        final_vector = attention_scores @ values
        
        final_vector = np.transpose(final_vector, axes=[0, 2, 1, 3])
        final_vector = np.reshape(final_vector, (B, T, self.dim_out))
        final_vector = self.final_projection.forward(final_vector)

        self.queries_matrix = queries
        self.keys_matrix = keys
        self.values_matrix = values
        self.attention_weights = attention_weights

class LayerNorm:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.gain = np.ones((1, input_dim))
        self.bais = np.zeros((1, input_dim))
        
        self.mean = None
        self.mu = None
        self.var = None
        self.std = None
        self.out = None
        
        self.grad_gain = np.zeros(self.gain.shape)
        self.grad_bais = np.zeros(self.bais.shape)
        
    def forward(self, inputs):
        epsilon = 1e-5
        self.mean = (1 / self.input_dim) * np.sum(inputs, axis=-1, keepdims=True)
        self.mu = inputs - self.mean
        self.var = (1 / self.input_dim) * np.sum(np.square(self.mu))
        self.std = np.sqrt(self.var + epsilon)
        self.out = self.gain * (self.mu / self.std) + self.bais
        
        return self.out
    
    def backward(self, prev_grad):
        direct_path = prev_grad * (self.gain / self.std)
        indirect_path = np.sum(
            (prev_grad[..., np.newaxis] * 1/(self.input_dim) * (self.gain[..., np.newaxis]) * 
             1/(self.std[..., np.newaxis]) + (np.expand_dims(self.mu, axis=-1) * np.expand_dims(self.mu, axis=-2))/(np.power(se, axis=-2)))
        )
        dL_dX = direct_path - indirect_path
        dL_dGain = np.sum(np.sum(prev_grad * (self.mu / self.std), axis=0, keepdims=True), axis=1)
    
        dL_dB = np.sum(np.sum(prev_grad, axis=0, keepdims=True), axis=1)
        
        
        self.grad_gain = dL_dGain
        self.grad_bais = dL_dB
        
        return dL_dX