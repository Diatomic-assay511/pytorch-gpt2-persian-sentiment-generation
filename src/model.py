"""
From-scratch implementation of the GPT-2 model architecture.

This includes:
- CausalSelfAttention
- MLP (Feed-Forward Network)
- Block (Transformer Decoder Block)
- GPT2 (The full model)
"""

import math
import logging
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import GPT2Config

logger = logging.getLogger(__name__)

class CausalSelfAttention(nn.Module):
    """
    A from-scratch implementation of Masked Multi-Head Self-Attention
    as used in GPT-2.
    """
    def __init__(self, config: GPT2Config):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError(
                f"Embedding dim ({config.n_embd}) must be divisible "
                f"by num_heads ({config.n_head})"
            )
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # Single linear layer for Q, K, V projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        
        # Output projection layer
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        # Causal mask
        # self.register_buffer registers a tensor that is part of the model's
        # state, but not a parameter (not trained).
        # We create a lower-triangular matrix of ones.
        mask = torch.tril(torch.ones(config.n_positions, config.n_positions))
        self.register_buffer(
            "mask", 
            mask.view(1, 1, config.n_positions, config.n_positions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Causal Self-Attention.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C)
                              (Batch, SequenceLength, EmbeddingDim)

        Returns:
            torch.Tensor: Output tensor of shape (B, T, C)
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dim

        # 1. Calculate Q, K, V from input x
        qkv = self.c_attn(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.n_embd, dim=2)  # (B, T, C) each

        # 2. Reshape for multi-head attention
        # (B, T, C) -> (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 3. Compute attention scores (Scaled Dot-Product)
        # (B, n_head, T, head_dim) @ (B, n_head, head_dim, T) -> (B, n_head, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 4. Apply causal mask
        # (B, n_head, T, T)
        # self.mask is (1, 1, max_T, max_T)
        # We slice it to match the current sequence length T
        causal_mask = self.mask[:, :, :T, :T]
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        
        # 5. Apply softmax to get attention weights
        att = F.softmax(scores, dim=-1)  # (B, n_head, T, T)
        att = self.attn_dropout(att)
        
        # 6. Apply attention weights to values
        # (B, n_head, T, T) @ (B, n_head, T, head_dim) -> (B, n_head, T, head_dim)
        y = torch.matmul(att, v)
        
        # 7. Concatenate heads back together
        # (B, n_head, T, head_dim) -> (B, T, n_head, head_dim) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # 8. Apply output projection and residual dropout
        y = self.resid_dropout(self.c_proj(y))
        
        return y

class MLP(nn.Module):
    """
    The two-layer Feed-Forward Network (MLP) from the Transformer,
    using the GELU activation function.
    """
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_inner)
        self.c_proj = nn.Linear(config.n_inner, config.n_embd)
        self.dropout = nn.Dropout(config.resid_pdrop)
        
        # Use the 'tanh' approximation for GELU to match the original GPT-2
        # F.gelu is cleaner than the manual formula.
        self.activation = lambda x: F.gelu(x, approximate='tanh')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C)

        Returns:
            torch.Tensor: Output tensor of shape (B, T, C)
        """
        # (B, T, C) -> (B, T, C_inner) -> (B, T, C)
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """
    A single Transformer Decoder Block (GPT-2 style).
    Uses pre-LayerNorm (LayerNorm is applied *before* sub-modules).
    """
    def __init__(self, config: GPT2Config):
        super().__init__()
        # Layer normalization before attention
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # Multi-head self-attention
        self.attn = CausalSelfAttention(config)
        
        # Layer normalization before MLP
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # Feed-forward network (MLP)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Transformer Block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C)

        Returns:
            torch.Tensor: Output tensor of shape (B, T, C)
        """
        # First residual connection: x + attention(layer_norm(x))
        x = x + self.attn(self.ln_1(x))
        
        # Second residual connection: x + mlp(layer_norm(x))
        x = x + self.mlp(self.ln_2(x))
        
        return x

class GPT2(nn.Module):
    """
    The full GPT-2 model architecture (scaled-down).
    """
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict({
            # Token Embeddings
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            # Position Embeddings (learnable)
            'wpe': nn.Embedding(config.n_positions, config.n_embd),
            # Embedding dropout
            'drop': nn.Dropout(config.embd_pdrop),
            # Transformer blocks
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # Final layer normalization
            'ln_f': nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        })

        # Language model head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Tie weights between token embeddings and the final head
        # This is a common practice for language models.
        self.transformer['wte'].weight = self.lm_head.weight
        logger.info("Tied weights between wte and lm_head.")

        # Apply custom weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """
        Initializes weights according to the GPT-2 paper.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Initialize weights from a normal dist
            torch.nn.init.normal_(
                module.weight, 
                mean=0.0, 
                std=self.config.initializer_range
            )
            # Initialize bias to zero if it exists
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            # Initialize LayerNorm bias to zero and weight to ones
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the GPT-2 model.

        Args:
            input_ids (torch.Tensor): Shape (B, T)
            attention_mask (torch.Tensor, optional): Shape (B, T). Not strictly
                used by this GPT-2 implementation (as CausalSelfAttention
                handles masking), but kept for API compatibility.
            labels (torch.Tensor, optional): Shape (B, T). If provided,
                the model calculates and returns the loss.

        Returns:
            Dict[str, torch.Tensor]: A dictionary with 'logits' and 'loss'.
        """
        B, T = input_ids.size()  # batch size, sequence length
        if T > self.config.n_positions:
            raise ValueError(
                f"Sequence length ({T}) exceeds model's max positions "
                f"({self.config.n_positions})."
            )

        device = input_ids.device
        
        # 1. Create position indices (0, 1, ..., T-1)
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

        # 2. Compute embeddings
        # Token embeddings
        tok_emb = self.transformer['wte'](input_ids)  # (B, T, n_embd)
        # Position embeddings
        pos_emb = self.transformer['wpe'](pos)  # (1, T, n_embd)
        
        # Combine embeddings and apply dropout
        x = self.transformer['drop'](tok_emb + pos_emb)  # (B, T, n_embd)
        
        # 3. Process through transformer blocks
        for block in self.transformer['h']:
            x = block(x)  # (B, T, n_embd)
            
        # 4. Apply final layer normalization
        x = self.transformer['ln_f'](x)  # (B, T, n_embd)
        
        # 5. Compute logits using the language model head
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # 6. Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next token prediction
            # We want to predict token[i+1] using token[i]
            # Logits: (B, T, V) -> (B, T-1, V) (remove last token's logit)
            # Labels: (B, T)   -> (B, T-1)   (remove first token's label)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten for cross entropy calculation
            # (B, T-1, V) -> (B*(T-1), V)
            # (B, T-1)    -> (B*(T-1))
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1),
                ignore_index=-100 # Standard ignore index for padding
            )
            
        return {'logits': logits, 'loss': loss}

    @torch.no_grad()
    def generate(
        self, 
        input_ids: torch.Tensor, 
        max_new_tokens: int,
        temperature: float = 1.0, 
        top_k: int = 0, 
        top_p: float = 1.0
    ) -> torch.Tensor:
        """
        Generates text auto-regressively with sampling.

        Args:
            input_ids (torch.Tensor): The prompt, shape (B, T_prompt).
            max_new_tokens (int): The number of new tokens to generate.
            temperature (float): Softmax temperature. Lower is more deterministic.
            top_k (int): If > 0, sample from the k most likely tokens.
            top_p (float): If < 1.0, use nucleus sampling.

        Returns:
            torch.Tensor: The generated sequence, shape (B, T_prompt + max_new_tokens).
        """
        # 1. Set model to evaluation mode
        self.eval()
        
        for _ in range(max_new_tokens):
            # 2. Truncate sequence if it exceeds model's context window
            # We only need the last n_positions tokens to predict the next one
            if input_ids.size(1) > self.config.n_positions:
                input_ids_cond = input_ids[:, -self.config.n_positions:]
            else:
                input_ids_cond = input_ids
            
            # 3. Run forward pass to get logits for the *last* token
            outputs = self(input_ids_cond)
            logits = outputs['logits'][:, -1, :]  # (B, vocab_size)
            
            # 4. Apply temperature scaling
            logits = logits / temperature
            
            # 5. Apply Top-K sampling (if k > 0)
            if top_k > 0:
                # Get the top k values and their indices
                top_k_values, top_k_indices = torch.topk(logits, top_k, dim=-1)
                # Set all other logits to negative infinity
                logits_filtered = torch.full_like(logits, float('-inf'))
                logits_filtered.scatter_(-1, top_k_indices, top_k_values)
                logits = logits_filtered
            
            # 6. Apply Nucleus (Top-P) sampling (if p < 1.0)
            if top_p < 1.0:
                # Sort logits in descending order
                sorted_logits, sorted_indices = torch.sort(
                    logits, descending=True, dim=-1
                )
                # Compute cumulative probabilities
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                
                # Find indices to remove (those *after* the cumulative prob > top_p)
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift right: we want to *keep* the first token that exceeds p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Create mask for original indices
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            # 7. Sample the next token
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            
            # 8. Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # 9. Stop if EOS token is generated
            if self.config.eos_token_id is not None and \
               (next_token == self.config.eos_token_id).all():
                break

        return input_ids
