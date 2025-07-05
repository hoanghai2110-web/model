
import json
from transformers import AutoTokenizer

def validate_dataset():
    print("🔍 Validating Dataset Format...")
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("./qwen2_tokenizer", local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"✅ Tokenizer loaded: vocab_size = {tokenizer.vocab_size}")
    except Exception as e:
        print(f"❌ Cannot load tokenizer: {e}")
        return
    
    # Test basic tokenization
    print("\n🧪 Testing Basic Tokenization:")
    test_text = "<|im_start|>user\nBạn là ai?<|im_end|>"
    
    # Encode
    encoded = tokenizer(test_text, add_special_tokens=False)
    input_ids = encoded["input_ids"]
    print(f"Original: {test_text}")
    print(f"Token IDs: {input_ids}")
    
    # Decode
    decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
    print(f"Decoded: {decoded}")
    
    # Check if round-trip works
    if decoded == test_text:
        print("✅ Round-trip tokenization PASSED")
    else:
        print("❌ Round-trip tokenization FAILED")
        print(f"Expected: {repr(test_text)}")
        print(f"Got:      {repr(decoded)}")
        return False
    
    # Check dataset format
    print("\n📊 Checking Dataset Format:")
    valid_samples = 0
    total_samples = 0
    
    with open("dataset.jsonl", 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
                
            total_samples += 1
            try:
                data = json.loads(line)
                
                if 'messages' in data:
                    # Convert to Qwen2 format
                    user_msg = None
                    assistant_msg = None
                    
                    for msg in data['messages']:
                        if msg['role'] == 'user':
                            user_msg = msg['content'].strip()
                        elif msg['role'] == 'assistant':
                            assistant_msg = msg['content'].strip()
                    
                    if user_msg and assistant_msg:
                        # Create correct Qwen2 format
                        correct_text = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>"
                        
                        # Test tokenization of this sample
                        sample_ids = tokenizer(correct_text, add_special_tokens=False)["input_ids"]
                        sample_decoded = tokenizer.decode(sample_ids, skip_special_tokens=False)
                        
                        if sample_decoded == correct_text:
                            valid_samples += 1
                            if total_samples <= 3:  # Show first 3 examples
                                print(f"Sample {total_samples}: ✅")
                                print(f"  Text: {correct_text[:100]}...")
                                print(f"  Tokens: {len(sample_ids)}")
                        else:
                            print(f"Sample {total_samples}: ❌ Tokenization failed")
                            if total_samples <= 3:
                                print(f"  Expected: {repr(correct_text[:50])}...")
                                print(f"  Got:      {repr(sample_decoded[:50])}...")
                
            except Exception as e:
                print(f"❌ Error parsing line {line_num}: {e}")
    
    print(f"\n📈 Dataset Validation Results:")
    print(f"  Total samples: {total_samples}")
    print(f"  Valid samples: {valid_samples}")
    print(f"  Success rate: {valid_samples/total_samples*100:.1f}%")
    
    if valid_samples < total_samples * 0.9:
        print("⚠️  WARNING: Low success rate! Check dataset format.")
        return False
    
    print("✅ Dataset format validation PASSED")
    return True

if __name__ == "__main__":
    validate_dataset()
