from tqdm import tqdm
import urllib.request
import ssl
import pickle
import os

from data import Word2VecDataset
from model import Word2VecSGNS

def download_text_data():
    """
    Downloads the 'Tiny Shakespeare' dataset.
    """
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    print(f"Downloading data from: {url}...")
    
    context = ssl._create_unverified_context()
    response = urllib.request.urlopen(url, context=context)
    text = response.read().decode('utf-8')
    print(f"Downloaded {len(text)} characters.")
    return text

def main():
    text_data = download_text_data()
    
    WINDOW_SIZE = 2
    NUM_NEGATIVES = 5
    MIN_COUNT = 5
    EMBED_DIM = 50
    BATCH_SIZE = 128
    EPOCHS = 5
    INITIAL_LR = 1e-2

    dataset = Word2VecDataset(
        text=text_data, 
        window_size=WINDOW_SIZE, 
        num_negatives=NUM_NEGATIVES, 
        min_count=MIN_COUNT
    )
    
    model = Word2VecSGNS(
        vocab_size=dataset.vocab_size, 
        embed_dim=EMBED_DIM, 
        learning_rate=INITIAL_LR
    )

    print("\nStarting training...")
    for epoch in range(EPOCHS):
        total_loss = 0.0
        batch_count = 0
        
        batches = dataset.generate_batches(BATCH_SIZE)
        approx_total_batches = len(dataset.data) // BATCH_SIZE
        
        progress_bar = tqdm(batches, total=approx_total_batches, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for target_ids, context_ids, negative_ids in progress_bar:
            loss = model.step(target_ids, context_ids, negative_ids)
            total_loss += loss
            batch_count += 1
            
            if batch_count % 10 == 0:
                progress_bar.set_postfix({'loss': f"{loss:.4f}"})
                
        model.learning_rate = INITIAL_LR * (1.0 - (epoch + 1) / EPOCHS)
        if model.learning_rate < 1e-4:
            model.learning_rate = 1e-4
            
        avg_loss = total_loss / max(1, batch_count)
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")

    print("\nSaving model...")
    model_data = {
        'W_in': model.W_in,
        'word2id': dataset.word2id,
        'id2word': dataset.id2word
    }
    
    os.makedirs('models', exist_ok=True)
    save_path = 'models/word2vec_shakespeare.pkl'
    
    with open(save_path, 'wb') as f:
        pickle.dump(model_data, f)
        
    print(f"Model successfully saved to: {save_path}")

if __name__ == "__main__":
    main()