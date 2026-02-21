import numpy as np
from collections import Counter
import re

class Word2VecDataset:
    def __init__(self, text, window_size=2, num_negatives=5, min_count=2):
        """
        Class for preparing data for the Word2Vec model.
        
        :param text: Raw input text (string).
        :param window_size: The size of the context window (left and right).
        :param num_negatives: Number of negative samples per positive word.
        :param min_count: Minimum word frequency to be included in the vocabulary.
        """
        self.window_size = window_size
        self.num_negatives = num_negatives
        
        tokens = self._tokenize(text)
        
        self.word2id, self.id2word, self.word_counts = self._build_vocab(tokens, min_count)
        self.vocab_size = len(self.word2id)

        print(f"Vocabulary size: {self.vocab_size} words.")
        
        self.data = [self.word2id[w] for w in tokens if w in self.word2id]
        
        self.neg_sample_probs = self._init_negative_sampling_probs()
        
    def _tokenize(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.split()

    def _build_vocab(self, tokens, min_count):
        word_counts = Counter(tokens)
        
        word_counts = {w: c for w, c in word_counts.items() if c >= min_count}
        word2id = {w: i for i, w in enumerate(word_counts.keys())}
        id2word = {i: w for w, i in word2id.items()}
        
        return word2id, id2word, word_counts

    def _init_negative_sampling_probs(self):
        """ 
        P(w) = count(w)^0.75 / sum(count^0.75).
        """
        counts = np.array([self.word_counts[self.id2word[i]] for i in range(self.vocab_size)])
        counts_pow = np.power(counts, 0.75)
        probs = counts_pow / np.sum(counts_pow)
        return probs

    def generate_batches(self, batch_size):
        """
        Yields: (target_ids, context_ids, negative_ids)
        """
        targets = []
        contexts = []
        
        for i, target_id in enumerate(self.data):
            start = max(0, i - self.window_size)
            end = min(len(self.data), i + self.window_size + 1)
            
            for j in range(start, end):
                if i != j:
                    context_id = self.data[j]
                    targets.append(target_id)
                    contexts.append(context_id)
                    
                    if len(targets) == batch_size:
                        negatives = np.random.choice(
                            self.vocab_size, 
                            size=(batch_size, self.num_negatives), 
                            p=self.neg_sample_probs, 
                            replace=True
                        )
                        
                        yield np.array(targets), np.array(contexts), negatives
                        
                        targets = []
                        contexts = []
                        
        if len(targets) > 0:
            current_batch_size = len(targets)
            negatives = np.random.choice(
                self.vocab_size, 
                size=(current_batch_size, self.num_negatives), 
                p=self.neg_sample_probs, 
                replace=True
            )
            yield np.array(targets), np.array(contexts), negatives