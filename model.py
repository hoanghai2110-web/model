
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers import AutoTokenizer

@dataclass
class TransformerLiteConfig:
    """Cấu hình cho Transformer Lite 130M - Tương thích Qwen2"""
    vocab_size: int = 151643  # Qwen2 tokenizer vocab size (actual)
    hidden_size: int = 768   # Embedding dimension
    num_layers: int = 12     # Số lớp transformer
    num_heads: int = 12      # Số attention heads
    intermediate_size: int = 2048  # FFN hidden dimension
    max_seq_len: int = 4096  # Mistral context length
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    rms_norm: bool = True    # Mistral sử dụng RMSNorm thay LayerNorm
    
    # Kỹ thuật tối ưu
    use_rope: bool = True           # Rotary Position Embedding
    use_swiglu: bool = True         # SwiGLU activation
    use_layer_scale: bool = True    # LayerScale cho stability
    use_weight_tying: bool = True   # Weight tying giữa embed và output
    use_gau: bool = True            # Gated Attention Unit (nhẹ hơn MHA)
    
    # LoRA configuration
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: list = None
    
    # Tham số LayerScale
    layer_scale_init: float = 1e-4
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = ["qkv_proj", "out_proj", "gate_proj", "up_proj", "down_proj"]
        # Tính toán tổng tham số
        self.total_params = self.estimate_params()
        
    def estimate_params(self) -> int:
        """Ước tính tổng số tham số"""
        # Embedding layers
        embed_params = self.vocab_size * self.hidden_size
        
        # Transformer layers
        attention_params = 4 * self.hidden_size * self.hidden_size  # Q,K,V,O projections
        ffn_params = 3 * self.hidden_size * self.intermediate_size  # SwiGLU: gate, up, down
        layer_norm_params = 2 * self.hidden_size  # Pre-attention & pre-ffn LayerNorm
        
        transformer_params = self.num_layers * (attention_params + ffn_params + layer_norm_params)
        
        # Output layer (nếu không weight tying)
        if not self.use_weight_tying:
            output_params = self.vocab_size * self.hidden_size
        else:
            output_params = 0
            
        total = embed_params + transformer_params + output_params
        
        # LoRA params (much smaller)
        if self.use_lora:
            lora_params = 0
            for module in self.lora_target_modules:
                if "qkv_proj" in module:
                    lora_params += 2 * self.lora_rank * self.hidden_size * 3  # Q, K, V
                elif "out_proj" in module:
                    lora_params += 2 * self.lora_rank * self.hidden_size
                elif module in ["gate_proj", "up_proj"]:
                    lora_params += 2 * self.lora_rank * self.intermediate_size
                elif "down_proj" in module:
                    lora_params += 2 * self.lora_rank * self.hidden_size
            total += lora_params * self.num_layers
            
        return total

class LoRALinear(nn.Module):
    """LoRA Linear layer for efficient fine-tuning"""
    
    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: int = 32, dropout: float = 0.1, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Original frozen weights
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None
        
        # LoRA weights
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout)
        
        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # Freeze original weights
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward pass
        result = F.linear(x, self.weight, self.bias)
        
        # LoRA forward pass
        lora_result = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        result += lora_result * self.scaling
        
        return result

class RMSNorm(nn.Module):
    """RMS Normalization - Như trong Mistral"""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class RoPEEmbedding(nn.Module):
    """Rotary Position Embedding - Tối ưu cho ngữ cảnh dài"""
    
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Tạo frequencies cho dim//2 (RoPE chỉ áp dụng cho một nửa dimension)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache sin/cos values
        self._cached_seq_len = 0
        self._cached_cos = None
        self._cached_sin = None
        
    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self._cached_seq_len:
            self._cached_seq_len = seq_len
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            self._cached_cos = torch.cos(freqs).to(device)
            self._cached_sin = torch.sin(freqs).to(device)
            
        return self._cached_cos[:seq_len], self._cached_sin[:seq_len]

def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Áp dụng RoPE cho Q và K"""
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    # q, k shape: [batch, heads, seq_len, head_dim]
    # cos, sin shape: [seq_len, head_dim//2]
    batch_size, num_heads, seq_len, head_dim = q.shape
    
    # Expand cos, sin to match q, k dimensions
    cos = cos[:seq_len]  # [seq_len, head_dim//2]
    sin = sin[:seq_len]  # [seq_len, head_dim//2]
    
    # Expand to full head_dim by repeating
    cos = torch.cat([cos, cos], dim=-1)  # [seq_len, head_dim]
    sin = torch.cat([sin, sin], dim=-1)  # [seq_len, head_dim]
    
    # Expand for batch and heads
    cos = cos.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, seq_len, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, seq_len, head_dim)
    
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot

class GatedAttentionUnit(nn.Module):
    """Gated Attention Unit với LoRA support"""
    
    def __init__(self, config: TransformerLiteConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        # Gated projections với LoRA
        if config.use_lora and "qkv_proj" in config.lora_target_modules:
            self.qkv_proj = LoRALinear(
                self.hidden_size, 3 * self.hidden_size, 
                rank=config.lora_rank, alpha=config.lora_alpha, 
                dropout=config.lora_dropout, bias=False
            )
        else:
            self.qkv_proj = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=False)
            
        self.gate_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        if config.use_lora and "out_proj" in config.lora_target_modules:
            self.out_proj = LoRALinear(
                self.hidden_size, self.hidden_size,
                rank=config.lora_rank, alpha=config.lora_alpha,
                dropout=config.lora_dropout, bias=False
            )
        else:
            self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # RoPE
        if config.use_rope:
            self.rope = RoPEEmbedding(self.head_dim, config.max_seq_len)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Projections
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        gate = torch.sigmoid(self.gate_proj(x))
        
        # Reshape for multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        if self.config.use_rope:
            cos, sin = self.rope(seq_len, x.device)
            q, k = apply_rope(q, k, cos, sin)
        
        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # Gated output
        output = self.out_proj(attn_output) * gate
        
        return output

class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network với LoRA support"""
    
    def __init__(self, config: TransformerLiteConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # SwiGLU projections với LoRA
        if config.use_lora and "gate_proj" in config.lora_target_modules:
            self.gate_proj = LoRALinear(
                self.hidden_size, self.intermediate_size,
                rank=config.lora_rank, alpha=config.lora_alpha,
                dropout=config.lora_dropout, bias=False
            )
        else:
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            
        if config.use_lora and "up_proj" in config.lora_target_modules:
            self.up_proj = LoRALinear(
                self.hidden_size, self.intermediate_size,
                rank=config.lora_rank, alpha=config.lora_alpha,
                dropout=config.lora_dropout, bias=False
            )
        else:
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            
        if config.use_lora and "down_proj" in config.lora_target_modules:
            self.down_proj = LoRALinear(
                self.intermediate_size, self.hidden_size,
                rank=config.lora_rank, alpha=config.lora_alpha,
                dropout=config.lora_dropout, bias=False
            )
        else:
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))  # SiLU activation
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class TransformerLiteBlock(nn.Module):
    """Một khối Transformer với LoRA support"""
    
    def __init__(self, config: TransformerLiteConfig):
        super().__init__()
        self.config = config
        
        # Normalization layers (RMSNorm hoặc LayerNorm)
        if config.rms_norm:
            self.ln1 = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.ln2 = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Attention và FFN
        if config.use_gau:
            self.attention = GatedAttentionUnit(config)
        else:
            self.attention = nn.MultiheadAttention(
                config.hidden_size, config.num_heads, 
                dropout=config.dropout, batch_first=True
            )
            
        self.ffn = SwiGLUFFN(config)
        
        # LayerScale cho training stability
        if config.use_layer_scale:
            self.layer_scale1 = nn.Parameter(torch.ones(config.hidden_size) * config.layer_scale_init)
            self.layer_scale2 = nn.Parameter(torch.ones(config.hidden_size) * config.layer_scale_init)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN Attention
        residual = x
        x = self.ln1(x)
        
        if self.config.use_gau:
            attn_output = self.attention(x, attention_mask)
        else:
            attn_output, _ = self.attention(x, x, x, attn_mask=attention_mask)
        
        if self.config.use_layer_scale:
            attn_output = attn_output * self.layer_scale1
            
        x = residual + self.dropout(attn_output)
        
        # Pre-LN FFN
        residual = x
        x = self.ln2(x)
        ffn_output = self.ffn(x)
        
        if self.config.use_layer_scale:
            ffn_output = ffn_output * self.layer_scale2
            
        x = residual + self.dropout(ffn_output)
        
        return x

class TransformerLite(nn.Module):
    """Transformer Lite 130M với LoRA support"""
    
    def __init__(self, config: TransformerLiteConfig):
        super().__init__()
        self.config = config
        
        # Embedding layers
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerLiteBlock(config) for _ in range(config.num_layers)
        ])
        
        # Final normalization
        if config.rms_norm:
            self.ln_f = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Output layer
        if config.use_weight_tying:
            self.lm_head = None  # Sẽ dùng token_embedding.weight
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Khởi tạo weights
        self.apply(self._init_weights)
        
        # Freeze non-LoRA parameters if using LoRA
        if config.use_lora:
            self.freeze_non_lora_params()
        
        print(f"Transformer Lite initialized with {self.count_parameters():,} parameters")
        if config.use_lora:
            print(f"Trainable LoRA parameters: {self.count_lora_parameters():,}")
    
    def _init_weights(self, module):
        """Khởi tạo weights theo best practices - Cải thiện cho conversation"""
        if isinstance(module, nn.Linear):
            # ✅ Sử dụng Xavier uniform cho stable training
            torch.nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # ✅ Normal init với std nhỏ hơn cho embedding
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def freeze_non_lora_params(self):
        """Freeze tất cả parameters trừ LoRA"""
        for name, param in self.named_parameters():
            if 'lora_' not in name:
                param.requires_grad = False
        
        print("Frozen non-LoRA parameters for efficient fine-tuning")
    
    def count_parameters(self) -> int:
        """Đếm tổng số parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def count_lora_parameters(self) -> int:
        """Đếm số parameters LoRA có thể train"""
        return sum(p.numel() for name, p in self.named_parameters() if 'lora_' in name and p.requires_grad)
    
    def load_qwen2_tokenizer(self, tokenizer_path: str = "./qwen2_tokenizer"):
        """Load Qwen2 tokenizer từ folder local"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
            
            # Setup pad token để tránh warnings
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            
            print(f"Loaded Qwen2 tokenizer from {tokenizer_path}")
            print(f"Vocab size: {tokenizer.vocab_size}")
            print(f"Model vocab size: {self.config.vocab_size}")
            
            # Test tokenizer với tiếng Việt
            test_vi = "Xin chào, bạn khỏe không?"
            tokens = tokenizer.encode(test_vi)
            decoded = tokenizer.decode(tokens)
            print(f"Vietnamese test - Original: {test_vi}")
            print(f"Vietnamese test - Decoded: {decoded}")
            
            return tokenizer
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            return None
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        x = self.token_embedding(input_ids)
        
        # Prepare attention mask
        if attention_mask is not None:
            # Convert to causal mask
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.expand(batch_size, 1, seq_len, seq_len)
            
            # Causal mask
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device))
            attention_mask = attention_mask * causal_mask
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)
        
        # Final LayerNorm
        x = self.ln_f(x)
        
        # Output projection
        if self.config.use_weight_tying:
            logits = F.linear(x, self.token_embedding.weight)
        else:
            logits = self.lm_head(x)
        
        return logits
    
    def generate(self, tokenizer, prompt: str, max_length: int = 100, temperature: float = 0.8, top_p: float = 0.9):
        """Generate text với nucleus sampling"""
        self.eval()
        
        # Tokenize input
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                outputs = self(inputs)
                logits = outputs[:, -1, :] / temperature
                
                # Nucleus sampling
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
                
                # Sample from the filtered distribution
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Add to sequence
                inputs = torch.cat([inputs, next_token], dim=-1)
                
                # Stop if EOS token
                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        # Decode
        response = tokenizer.decode(inputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip()

# Tạo configuration optimized cho fine-tuning
def create_transformer_lite_for_finetuning():
    """Tạo Transformer Lite với LoRA cho fine-tuning hiệu quả"""
    config = TransformerLiteConfig(
        vocab_size=32000,  # Mistral vocab size
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        intermediate_size=2048,
        max_seq_len=4096,
        dropout=0.1,
        rms_norm=True,
        use_rope=True,
        use_swiglu=True,
        use_layer_scale=True,
        use_weight_tying=True,
        use_gau=True,
        
        # LoRA configuration cho fine-tuning hiệu quả
        use_lora=True,
        lora_rank=32,          # ✅ Tăng rank để model học được nhiều hơn
        lora_alpha=64,         # ✅ Tăng alpha = 2 * rank để scaling mạnh hơn
        lora_dropout=0.05,     # ✅ Giảm dropout để tránh under-fitting
        lora_target_modules=["qkv_proj", "out_proj", "gate_proj", "up_proj", "down_proj"],
        
        layer_scale_init=1e-4
    )
    
    model = TransformerLite(config)
    return model, config

def setup_for_conversation_finetuning(use_pretrained=False):
    """Setup cho fine-tuning conversation"""
    # Load tokenizer trước
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained("./qwen2_tokenizer", local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"Loaded Qwen2 tokenizer with vocab_size: {tokenizer.vocab_size}")
    except:
        print("Could not load local tokenizer, using remote")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    # ✅ Option to load pretrained weights instead of random
    if use_pretrained:
        print("🔄 Loading pretrained Qwen2-0.5B model...")
        try:
            from transformers import AutoModelForCausalLM
            pretrained_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")
            print("✅ Loaded pretrained Qwen2-0.5B successfully!")
            return pretrained_model, tokenizer, None, None, pretrained_model.config
        except Exception as e:
            print(f"❌ Failed to load pretrained model: {e}")
            print("📝 Falling back to custom TransformerLite...")
    
    # ✅ Use EXACT tokenizer vocab_size instead of adding buffer
    config_vocab_size = tokenizer.vocab_size
    
    print(f"🔧 Using EXACT tokenizer vocab_size: {config_vocab_size}")
    
    # Validate all special tokens are within vocab_size
    special_tokens = {
        'eos': tokenizer.eos_token_id,
        'pad': tokenizer.pad_token_id,
        'unk': getattr(tokenizer, 'unk_token_id', None)
    }
    
    for name, token_id in special_tokens.items():
        if token_id is not None and token_id >= config_vocab_size:
            print(f"⚠️ WARNING: {name}_token_id ({token_id}) >= vocab_size ({config_vocab_size})")
            config_vocab_size = max(config_vocab_size, token_id + 1)
    
    print(f"🔧 Setting model vocab_size to {config_vocab_size}")
    print(f"   Tokenizer vocab_size: {tokenizer.vocab_size}")
    
    config = TransformerLiteConfig(
        vocab_size=config_vocab_size,  # ✅ Safe vocab size với buffer
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        intermediate_size=2048,
        max_seq_len=4096,
        dropout=0.1,
        rms_norm=True,
        use_rope=True,
        use_swiglu=True,
        use_layer_scale=True,
        use_weight_tying=True,
        use_gau=True,
        use_lora=True,
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.1,
        lora_target_modules=["qkv_proj", "out_proj", "gate_proj", "up_proj", "down_proj"],
        layer_scale_init=1e-4
    )
    
    # Tạo model với config đã fix
    model = TransformerLite(config)
    
    if tokenizer is None:
        print("Could not load tokenizer, using AutoTokenizer with remote fallback")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
    
    # Optimizer chỉ cho LoRA parameters - Giảm learning rate để học kỹ hơn
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, 
        lr=5e-5,  # Giảm LR từ 2e-4 xuống 5e-5 để học chậm hơn
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler - Tăng T_max cho 10 epochs
    # Ước tính số steps (sẽ được điều chỉnh trong training loop)
    estimated_steps = 1000  # Giá trị ước tính, sẽ được update sau
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=estimated_steps,
        eta_min=1e-6
    )
    
    print(f"Setup complete!")
    print(f"Total parameters: {model.count_parameters():,}")
    print(f"Trainable LoRA parameters: {model.count_lora_parameters():,}")
    print(f"LoRA efficiency: {model.count_lora_parameters() / model.count_parameters() * 100:.2f}%")
    
    return model, tokenizer, optimizer, scheduler, config

def create_conversation_dataset(conversations, tokenizer, max_length=512):
    """Tạo dataset cho conversation fine-tuning"""
    input_ids = []
    labels = []
    
    for conv in conversations:
        # Format: "Human: {question}\n\nAssistant: {answer}"
        text = f"Human: {conv['human']}\n\nAssistant: {conv['assistant']}"
        
        # Tokenize
        tokens = tokenizer.encode(text, max_length=max_length, truncation=True, return_tensors="pt")
        
        # Create labels (same as input_ids for causal LM)
        input_ids.append(tokens.squeeze())
        labels.append(tokens.squeeze())
    
    return input_ids, labels

# Test model
if __name__ == "__main__":
    # Setup cho fine-tuning
    model, tokenizer, optimizer, scheduler, config = setup_for_conversation_finetuning()
    
    # Test tokenizer
    if tokenizer:
        test_text = "Xin chào! Bạn có thể giúp tôi không?"
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        print(f"Test text: {test_text}")
        print(f"Tokens: {tokens}")
        print(f"Decoded: {decoded}")
    
    # Test forward pass
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Test generation
    if tokenizer:
        test_prompt = "Human: Xin chào!\n\nAssistant:"
        response = model.generate(tokenizer, test_prompt, max_length=50)
        print(f"Generated: {response}")
