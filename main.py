import numpy as np

class EmbeddingLayer:
    def __init__(self, vocab_size, embedding_dim):
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.02
        self.inputs = None
        
    def forward(self, inputs):
        self.inputs = inputs
        return self.embeddings[inputs]
    
    def backward(self, prev_grad):
        grad_embeddings = np.zeros(self.embeddings.shape)
        np.add.at(grad_embeddings, self.inputs, prev_grad)
        return grad_embeddings

    def get_params(self):
        return {'embeddings': self.embeddings}

class Linear:
    def __init__(self, dim_in, dim_out, include_bias=True):
        self.weights = np.random.randn(dim_in, dim_out) * np.sqrt(2.0 / (dim_in + dim_out))
        self.include_bias = include_bias
        self.bias = np.zeros((1, dim_out)) if include_bias else None
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
        if self.include_bias: params['b'] = self.bias
        return params

class Softmax:
    def __init__(self):
        self.softmax_out = None
        
    def forward(self, inputs):
        max_logit = np.max(inputs, axis=-1, keepdims=True)
        exps = np.exp(inputs - max_logit)
        self.softmax_out = exps / np.sum(exps, axis=-1, keepdims=True)
        return self.softmax_out
    
    def backward(self, prev_grad):
        return prev_grad * self.softmax_out - self.softmax_out * np.sum(self.softmax_out * prev_grad, axis=-1, keepdims=True)

class CrossEntropyLossWithSoftmax:
    def __init__(self):
        self.logits = None
        self.labels = None
        
    def forward(self, logits, labels):
        B, T, V = logits.shape
        logits_flat = logits.reshape(-1, V)
        labels_flat = labels.reshape(-1)
        max_logit = np.max(logits_flat, axis=-1, keepdims=True)
        logits_stable = logits_flat - max_logit
        lse = max_logit + np.log(np.sum(np.exp(logits_stable), axis=-1, keepdims=True))
        log_softmax = logits_flat[np.arange(B * T), labels_flat] - lse.flatten()
        self.logits = logits
        self.labels = labels
        return -np.sum(log_softmax) / (B * T)
    
    def backward(self):
        B, T, V = self.logits.shape
        logits_flat = self.logits.reshape(-1, V)
        labels_flat = self.labels.reshape(-1)
        max_logit = np.max(logits_flat, axis=-1, keepdims=True)
        exps = np.exp(logits_flat - max_logit)
        probs = exps / np.sum(exps, axis=-1, keepdims=True)
        probs[np.arange(B * T), labels_flat] -= 1
        grad = probs / (B * T)
        return grad.reshape(B, T, V)

class LayerNorm:
    def __init__(self, dim):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.eps = 1e-5
        self.inputs = None
        self.mean = None
        self.var = None
        self.gamma_grad = None
        self.beta_grad = None

    def forward(self, x):
        self.inputs = x
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        x_hat = (x - self.mean) / np.sqrt(self.var + self.eps)
        return self.gamma * x_hat + self.beta

    def backward(self, prev_grad):
        B, T, D = prev_grad.shape
        x_hat = (self.inputs - self.mean) / np.sqrt(self.var + self.eps)
        
        self.gamma_grad = np.sum(prev_grad * x_hat, axis=(0, 1))
        self.beta_grad = np.sum(prev_grad, axis=(0, 1))
        
        dx_hat = prev_grad * self.gamma
        dvar = np.sum(dx_hat * (self.inputs - self.mean) * -0.5 * (self.var + self.eps)**-1.5, axis=-1, keepdims=True)
        dmean = np.sum(dx_hat * -1 / np.sqrt(self.var + self.eps), axis=-1, keepdims=True) + \
                dvar * np.mean(-2 * (self.inputs - self.mean), axis=-1, keepdims=True)
        
        grad_inputs = dx_hat / np.sqrt(self.var + self.eps) + \
                      dvar * 2 * (self.inputs - self.mean) / (B * T * D) + \
                      dmean / (B * T)
        return grad_inputs

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x > 0)
        return np.where(self.mask, x, 0)

    def backward(self, prev_grad):
        return prev_grad * self.mask

class MultiHeadAttention:
    def __init__(self, dim_in, dim_out, n_heads):
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_heads = n_heads
        self.head_dim = dim_out // n_heads
        self.query_linear = Linear(dim_in, dim_out, include_bias=False)
        self.key_linear = Linear(dim_in, dim_out, include_bias=False)
        self.value_linear = Linear(dim_in, dim_out, include_bias=False)
        self.out_linear = Linear(dim_out, dim_out, include_bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query_linear.forward(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.key_linear.forward(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.value_linear.forward(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        mask = np.tril(np.ones((T, T)))
        scores = np.where(mask == 0, -1e9, scores)
        
        attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)
        
        context = np.matmul(attn_weights, v).transpose(0, 2, 1, 3).reshape(B, T, self.dim_out)
        return self.out_linear.forward(context)

    def backward(self, prev_grad):
        pass
