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
    
class Softax:
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

