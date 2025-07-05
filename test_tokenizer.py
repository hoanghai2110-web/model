
import torch
from transformers import AutoTokenizer

def test_qwen2_tokenizer():
    """Test Qwen2 tokenizer với tiếng Việt"""
    
    print("🧪 Testing Qwen2 Tokenizer")
    print("="*50)
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("./qwen2_tokenizer", local_files_only=True)
        print(f"✅ Loaded Qwen2 tokenizer")
        print(f"📊 Vocab size: {tokenizer.vocab_size}")
        print(f"🔤 Special tokens:")
        print(f"   EOS: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
        print(f"   PAD: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
        print(f"   UNK: {tokenizer.unk_token}")
        
        # Setup pad token nếu cần
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        return
    
    # Test với các câu tiếng Việt
    test_sentences = [
        "Xin chào, bạn khỏe không?",
        "Tôi yêu Việt Nam",
        "Hôm nay thời tiết đẹp quá!",
        "Bạn có thể giúp tôi không?",
        "Cảm ơn bạn rất nhiều"
    ]
    
    print(f"\n🇻🇳 Testing Vietnamese sentences:")
    print("-" * 50)
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n[{i}] Original: {sentence}")
        
        # Encode
        tokens = tokenizer.encode(sentence, add_special_tokens=True)
        print(f"    Tokens: {tokens[:10]}... (length: {len(tokens)})")
        
        # Decode
        decoded = tokenizer.decode(tokens, skip_special_tokens=True)
        print(f"    Decoded: {decoded}")
        
        # Check quality
        if decoded.strip() == sentence.strip():
            print(f"    ✅ Perfect match!")
        else:
            print(f"    ⚠️  Slight difference")
    
    # Test conversation format
    print(f"\n💬 Testing conversation format:")
    print("-" * 50)
    
    user_msg = "Bạn là ai?"
    assistant_msg = "Tôi là HyperMamba, trợ lý AI của bạn!"
    
    conversation = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>"
    
    print(f"Conversation format:")
    print(conversation)
    
    tokens = tokenizer.encode(conversation, add_special_tokens=True)
    decoded = tokenizer.decode(tokens, skip_special_tokens=False)
    
    print(f"\nTokens length: {len(tokens)}")
    print(f"Decoded:\n{decoded}")

if __name__ == "__main__":
    test_qwen2_tokenizer()
