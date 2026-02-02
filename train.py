import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from nano_llm.mini_transformer import MiniTransformer
from nano_llm.tokenizer import CharTokenizer
from nano_llm.dataset import TextDataset
from nano_llm.generation import generate # Asegúrate de que la ruta sea correcta

def train():
    """
    Main training loop and overfitting test.
    
    This script initializes the model, loads a single batch of data,
    and attempts to overfit the model to ensure gradients are flowing
    correctly and the loss decreases.
    """
    # --- 1. Configuration ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    block_size = 64  # Context window
    batch_size = 4
    learning_rate = 5e-4
    max_iters = 500  # Number of iterations for overfit test
    
    # --- 2. Data Preparation ---
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    tokenizer = CharTokenizer(text)
    dataset = TextDataset('input.txt', block_size, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # --- 3. Model Initialization ---
    model = MiniTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        num_heads=8,
        ff_hidden=1024,
        num_layers=6,
        max_len=block_size,
        dropout=0.0 # No dropout for overfitting test
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # --- 4. Overfitting Test (Single Batch) ---
    model.train()
    # Get just one batch and stay with it
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)

    print(f"Starting Overfit Test on {device}...")
    
    for i in range(max_iters):
        # Forward pass
        logits, _ = model(x)
        
        # Reshape for CrossEntropy: (B*T, Vocab)
        # B = Batch, T = Time (block_size), C = Vocab
        B, T, C = logits.shape
        loss = criterion(logits.view(B * T, C), y.view(B * T))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 50 == 0:
            print(f"Iteration {i:3d} | Loss: {loss.item():.6f}")

    # Save checkpoint after overfitting
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'vocab_size': tokenizer.vocab_size
    }
    torch.save(checkpoint, "overfit_check.pt")
    print("Test complete. Checkpoint saved as overfit_check.pt")

    # --- 5. Inference Test (Verification) ---
    model.eval()
    print("\n--- Model Generation Test (Overfitted sequence) ---")
    
    # Usamos el primer ejemplo del batch 'x' como semilla
    start_context = x[0:1, :10] # Tomamos los primeros 10 tokens del primer ejemplo
    
    generated_indices = generate(
        model=model,
        input_ids=start_context,
        max_new_tokens=block_size - 10,
        temperature=0.1 # Muy bajo para que sea determinista
    )
    
    result_text = tokenizer.decode(generated_indices[0].tolist())
    print(f"Generated text:\n{result_text}")
    
    # Comparamos con el original
    original_text = tokenizer.decode(x[0].tolist())
    print(f"\nExpected text (Original):\n{original_text}")

if __name__ == "__main__":
    train()