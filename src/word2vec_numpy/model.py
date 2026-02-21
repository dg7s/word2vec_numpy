import numpy as np

class Word2VecSGNS:
    def __init__(self, vocab_size, embed_dim=100, learning_rate=1e-3):
        """
        Word2Vec Skip-gram with Negative Sampling.
        
        :param vocab_size: Size of the vocabulary.
        :param embed_dim: Dimensionality of the word embeddings.
        :param learning_rate: Step size for Stochastic Gradient Descent.
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.learning_rate = learning_rate
        
        self.W_in = np.random.uniform(-0.5 / embed_dim, 0.5 / embed_dim, (vocab_size, embed_dim))
        
        self.W_out = np.zeros((vocab_size, embed_dim))

    def _sigmoid(self, x):
        res = np.zeros_like(x, dtype=np.float64)
        
        pos = x >= 0
        neg = ~pos
        
        res[pos] = 1 / (1 + np.exp(-x[pos]))
        res[neg] = np.exp(x[neg]) / (1 + np.exp(x[neg]))
        
        return res

    def step(self, target_ids, context_ids, negative_ids):
        """
        Performs one step of forward pass, calculates loss, 
        computes gradients, and updates weights.
        
        Shapes:
        - target_ids: (Batch Size,)
        - context_ids: (Batch Size,)
        - negative_ids: (Batch Size, Num Negatives)
        """
        batch_size = target_ids.shape[0]
        num_negatives = negative_ids.shape[1]

        # v_c: (B, D)
        v_c = self.W_in[target_ids]
        # u_p: (B, D)
        u_p = self.W_out[context_ids]
        # u_n: (B, K, D)
        u_n = self.W_out[negative_ids]

        # Positive scores
        pos_scores = np.sum(v_c * u_p, axis=1)  # Shape: (B,)
        
        # Negative scores: (B, 1, D) * (B, K, D) -> sum over D
        neg_scores = np.sum(v_c[:, np.newaxis, :] * u_n, axis=2)  # Shape: (B, K)

        pos_probs = self._sigmoid(pos_scores)
        neg_probs = self._sigmoid(neg_scores)

        # Loss = -log(sigmoid(pos_scores)) - sum(log(sigmoid(-neg_scores)))
        # sigmoid(-x) = 1 - sigmoid(x)
        pos_loss = -np.log(pos_probs + 1e-10)
        neg_loss = -np.sum(np.log(1.0 - neg_probs + 1e-10), axis=1)
        
        batch_loss = np.mean(pos_loss + neg_loss)

        # d_pos = sigmoid(score) - 1
        # d_neg = sigmoid(score) - 0
        d_pos = pos_probs - 1.0  # Shape: (B,)
        d_neg = neg_probs        # Shape: (B, K)

        # grad_W_out (context) = d_pos * v_c
        grad_u_p = d_pos[:, np.newaxis] * v_c  # Shape: (B, D)
        
        # grad_W_out (negatives) = d_neg * v_c
        grad_u_n = d_neg[:, :, np.newaxis] * v_c[:, np.newaxis, :]  # Shape: (B, K, D)
        
        # grad_W_in (target) = d_pos * u_p + sum(d_neg * u_n) over K
        grad_v_c = d_pos[:, np.newaxis] * u_p + np.sum(d_neg[:, :, np.newaxis] * u_n, axis=1)  # Shape: (B, D)

        # [Warning] Drops duplicate gradients
        self.W_in[target_ids] -= self.learning_rate * grad_v_c
        self.W_out[context_ids] -= self.learning_rate * grad_u_p
        
        neg_ids_flat = negative_ids.flatten()
        grad_u_n_flat = grad_u_n.reshape(-1, self.embed_dim)
        self.W_out[neg_ids_flat] -= self.learning_rate * grad_u_n_flat

        # np.add.at(self.W_in, target_ids, -self.learning_rate * grad_v_c)
        # np.add.at(self.W_out, neg_ids_flat, -self.learning_rate * grad_u_n_flat)
        # np.add.at(self.W_out, neg_ids_flat, -self.learning_rate * grad_u_n_flat)

        return batch_loss