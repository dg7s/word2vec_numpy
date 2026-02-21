import numpy as np
import pickle

def get_similar_words(word, word2id, id2word, W_in, top_n=5):
    """
    Finds the most similar words to a given word using Cosine Similarity.
    """
    if word not in word2id:
        return f"Word '{word}' not found in the vocabulary."
    
    word_id = word2id[word]
    word_vec = W_in[word_id]
    
    # Cosine Similarity: (A dot B) / (||A|| * ||B||)
    norms = np.linalg.norm(W_in, axis=1)
    word_norm = np.linalg.norm(word_vec)
    
    norms = np.where(norms == 0, 1e-10, norms)
    word_norm = word_norm if word_norm > 0 else 1e-10
    
    similarities = np.dot(W_in, word_vec) / (norms * word_norm)
    
    sorted_indices = np.argsort(similarities)
    top_indices_ascending = sorted_indices[-(top_n + 1):]
    closest_ids = top_indices_ascending[::-1]
    
    results = []
    for cid in closest_ids:
        if cid != word_id:
            results.append((id2word[cid], similarities[cid]))
            if len(results) == top_n:
                break
                
    return results

def main():
    model_path = 'models/word2vec_shakespeare.pkl'

    print("Loading model...")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
        
    W_in = model_data['W_in']
    word2id = model_data['word2id']
    id2word = model_data['id2word']
    
    print(f"Model loaded successfully! Vocabulary size: {len(word2id)} words.")
    print(f"Embedding dimension: {W_in.shape[1]}")
    
    test_words = ['king', 'queen', 'love', 'death', 'sword', 'man', 'lady']
    
    print("\n--- TESTING WORD SIMILARITY ---")
    for w in test_words:
        print(f"\nMost similar to '{w}':")
        sims = get_similar_words(w, word2id, id2word, W_in, top_n=4)
        
        if isinstance(sims, str):
            print(f"  {sims}")
        else:
            for sim_w, score in sims:
                print(f"  - {sim_w} (similarity: {score:.4f})")

if __name__ == "__main__":
    main()