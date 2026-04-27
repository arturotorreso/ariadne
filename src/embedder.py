import torch
import numpy as np

class SequenceEmbedder:
    def __init__(self, s=8, m=4, t=4, version="default", device=None):
        """
        Initializes the GPU-Accelerated deterministic RotorMap embedder.
        """
        self.s = s
        self.m = m
        self.t = t
        self.version = version
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.vocab = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0}
        self.embedding_dim = self.m * (4 ** self.t) * 2

        print(f"[Embedder] Initialized GPU-Accelerated RotorMap.")
        print(f"           s={s}, m={m}, t={t} -> Dim: {self.embedding_dim}")
        print(f"           Device: {self.device.type.upper()}")

        # Pre-compute the rotary angles (omegas) on the GPU
        self._init_omegas()

    def _init_omegas(self):
        """Pre-computes the rotation angles for the maximum expected chunk size."""
        # For our pipeline, chunks are 100bp. We pre-compute the rotations for speed.
        N = 100 
        k_values = torch.arange(1, self.m + 1, device=self.device, dtype=torch.float32)
        
        if self.version == "default" or self.m == 1:
            phases = k_values * (2 * np.pi / N)
        else:
            phases = ((2 * (k_values - 1) / (self.m - 1) + 1) * (2 * np.pi / N))
            
        # Store as complex numbers on the GPU
        self.omega_s = torch.polar(torch.ones_like(phases), phases)

    def embed_batch(self, sequences):
        """
        Executes the RotorMap algorithm across thousands of sequences simultaneously on the GPU.
        """
        batch_size = len(sequences)
        seq_len = len(sequences[0])
        
        # 1. Tokenize (CPU -> GPU)
        tokens = [[self.vocab.get(char, 0) for char in seq.upper()] for seq in sequences]
        dna = torch.tensor(tokens, dtype=torch.int32, device=self.device)
        
        # 2. Setup Masks and Output Matrix
        bmask = (1 << (2 * self.s)) - 1
        tbmask = (1 << (2 * self.t)) - 1
        mixer = 0x9E3779B1
        
        # Complex output matrix: (batch_size, m, 4^t)
        c = torch.zeros((batch_size, self.m, 4 ** self.t), dtype=torch.complex64, device=self.device)
        
        # 3. Read the first s-mer across the entire batch
        smer = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
        for i in range(self.s):
            smer <<= 2
            smer += dna[:, i]
            
        omega_i = torch.ones((batch_size, self.m), dtype=torch.complex64, device=self.device)
        
        # 4. Slide window and accumulate complex rotations (Fully Vectorized)
        for i in range(self.s, seq_len):
            lower = smer & tbmask
            upper = smer >> (2 * self.t)
            scramble_key = (upper * mixer) & tbmask
            tsmer = lower ^ scramble_key 
            
            # Scatter addition of complex rotations into the target tsmer bins
            # scatter_add_ requires expanding indices to match the m dimension
            tsmer_expanded = tsmer.unsqueeze(1).expand(-1, self.m).unsqueeze(-1).long()
            omega_expanded = omega_i.unsqueeze(-1)
            c.scatter_add_(2, tsmer_expanded, omega_expanded)
            
            smer <<= 2
            smer &= bmask
            smer += dna[:, i]
            omega_i *= self.omega_s
            
        # Process the final s-mer block
        lower = smer & tbmask
        upper = smer >> (2 * self.t)
        scramble_key = (upper * mixer) & tbmask
        tsmer = lower ^ scramble_key
        tsmer_expanded = tsmer.unsqueeze(1).expand(-1, self.m).unsqueeze(-1).long()
        omega_expanded = omega_i.unsqueeze(-1)
        c.scatter_add_(2, tsmer_expanded, omega_expanded)
        
        # 5. Normalize (L2 constraint mapping to Quantum Fidelity)
        norm = torch.sqrt(torch.sum(torch.abs(c)**2, dim=(1,2), keepdim=True) + 1e-16)
        c /= norm
        
        # 6. Flatten and stack Real/Imaginary parts so FAISS can calculate inner-products
        c_flat = c.reshape(batch_size, -1)
        embeddings = torch.cat((c_flat.real, c_flat.imag), dim=1)
        
        # RESTORED FIX: Force the CPU to wait until the GPU finishes the math
        # to prevent cross-read memory contamination!
        # if self.device.type == 'cuda':
        #     torch.cuda.synchronize(self.device)

        # Pull back to CPU for FAISS
        return embeddings.cpu().numpy().astype(np.float32)