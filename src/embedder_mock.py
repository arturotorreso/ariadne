import torch
import torch.nn as nn
import numpy as np

class MockRoPEModel(nn.Module):
    """
    A dummy model that mimics the I/O of your future RoPE Transformer.
    It takes an integer tensor of shape (batch_size, sequence_length) 
    and outputs a dense tensor of shape (batch_size, 512).
    """
    def __init__(self, embedding_dim=512):
        super().__init__()
        # Vocab size 6: A=0, C=1, G=2, T=3, N=4, PAD/Unknown=5
        self.embedding = nn.Embedding(num_embeddings=6, embedding_dim=embedding_dim)
        
    def forward(self, x):
        # x is (batch_size, seq_len)
        # embedded is (batch_size, seq_len, 512)
        embedded = self.embedding(x)
        # Mean pooling to squash the sequence length and return (batch_size, 512)
        pooled = embedded.mean(dim=1)
        return pooled

class SequenceEmbedder:
    def __init__(self, model, device='cuda'):
        """
        Initializes the wrapper. Forces the model into evaluation mode
        and moves it to the target device.
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval() # Disable dropout and batch norm layers
        
        # Fast lookup table for tokenization
        self.vocab = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}

    def _tokenize(self, sequences):
        """
        Translates raw strings into a 2D list of integers.
        """
        batch_tokens = []
        for seq in sequences:
            # Default to 5 (Unknown) if a weird character is found
            tokens = [self.vocab.get(char, 5) for char in seq.upper()]
            batch_tokens.append(tokens)
        return batch_tokens

    def embed_batch(self, sequences):
        """
        The critical inference block. Tokenizes, pushes to GPU, runs the model
        without gradients, and pulls the numpy array back to CPU.
        """
        # 1. Tokenize (CPU)
        tokens = self._tokenize(sequences)
        
        # 2. Convert to PyTorch Tensor and push to GPU VRAM
        input_tensor = torch.tensor(tokens, dtype=torch.long, device=self.device)
        
        # 3. Disable gradients to save VRAM and speed up math
        with torch.no_grad():
            output_tensor = self.model(input_tensor)
            
        # 4. Pull back to CPU RAM, convert to numpy, and enforce float32
        embeddings = output_tensor.cpu().numpy().astype(np.float32)
        
        return embeddings