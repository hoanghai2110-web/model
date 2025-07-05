
import torch
import os
import math
from model import setup_for_conversation_finetuning

def load_base_model(use_original_qwen=False):
    """Load base model để test - có thể chọn original Qwen2 hoặc custom model"""
    if use_original_qwen:
        print("🔄 Loading ORIGINAL Qwen2-0.6B model from Hugging Face...")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            
            print(f"✅ Original Qwen2 model loaded!")
            print(f"📊 Model vocab size: {model.config.vocab_size}")
            print(f"📊 Tokenizer vocab size: {tokenizer.vocab_size}")
            
            return model, tokenizer
        except Exception as e:
            print(f"❌ Error loading original model: {e}")
            print("📝 Falling back to custom model...")
    
    print("🔄 Loading CUSTOM base model (chưa fine-tune)...")
    model, tokenizer, _, _, config = setup_for_conversation_finetuning()
    
    # Fix tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"✅ Base model loaded!")
    print(f"📊 Model vocab size: {config.vocab_size}")
    print(f"📊 Tokenizer vocab size: {tokenizer.vocab_size}")
    
    return model, tokenizer

def test_qwen2_format_generation(model, tokenizer, user_input, max_length=100):
    """Test generation với Qwen2 format chuẩn"""
    
    # ✅ Sử dụng chat template chuẩn của Qwen2
    messages = [
        {"role": "system", "content": "Bạn là một trợ lý AI thân thiện và hiểu tiếng Việt."},
        {"role": "user", "content": user_input}
    ]
    
    # Apply chat template
    try:
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        print(f"🔍 Prompt đã format: {prompt}")
    except:
        # Fallback nếu không có chat template
        prompt = f"<|im_start|>system\nBạn là một trợ lý AI thân thiện và hiểu tiếng Việt.<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
        print(f"🔍 Fallback prompt: {prompt}")
    
    # Tokenize với validation
    try:
        inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
        input_ids = inputs['input_ids']
        
        # ✅ CRITICAL: Validate tokens không vượt quá model vocab size
        max_token_id = input_ids.max().item()
        model_vocab_size = model.config.vocab_size
        if max_token_id >= model_vocab_size:
            print(f"❌ ERROR: Token {max_token_id} >= model_vocab_size {model_vocab_size}")
            print(f"   Tokenizer vocab_size: {tokenizer.vocab_size}")
            return "❌ Lỗi tokenization"
        
        print(f"✅ Tokenization OK: max_token_id = {max_token_id}")
        
    except Exception as e:
        print(f"❌ Tokenization error: {e}")
        return "❌ Lỗi tokenization"
    
    model.eval()
    with torch.no_grad():
        generated_ids = input_ids.clone()
        
        for step in range(max_length):
            # Forward pass
            try:
                outputs = model(generated_ids)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            except Exception as e:
                print(f"❌ Forward pass error: {e}")
                break
            
            # ✅ Sử dụng temperature thấp hơn để tránh nonsense
            temperature = 0.3  # Giảm từ 0.8 xuống 0.3
            next_token_logits = logits[0, -1, :] / temperature
            
            # ✅ Top-k sampling nhỏ hơn + top-p
            top_k = 20  # Giảm từ 50 xuống 20
            top_k_logits, top_k_indices = torch.topk(next_token_logits, min(top_k, logits.size(-1)))
            
            # ✅ Áp dụng top-p (nucleus sampling)
            sorted_logits, sorted_indices = torch.sort(top_k_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # ✅ Chỉ giữ tokens có cumulative prob <= 0.8
            sorted_indices_to_remove = cumulative_probs > 0.8
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Map back to original indices
            indices_to_remove = torch.zeros_like(top_k_logits, dtype=torch.bool)
            indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)
            top_k_logits[indices_to_remove] = float('-inf')
            
            # Softmax và sample
            probs = torch.softmax(top_k_logits, dim=-1)
            
            # ✅ Nếu tất cả probs là 0, chọn token đầu tiên
            if torch.sum(probs) == 0:
                next_token_idx = torch.tensor([0])
            else:
                next_token_idx = torch.multinomial(probs, 1)
            
            next_token = top_k_indices[next_token_idx]
            
            # ✅ Validate token trước khi add
            if next_token.item() >= model.config.vocab_size:
                print(f"⚠️ Generated invalid token {next_token.item()}, stopping")
                break
            
            # ✅ Stop conditions
            if (next_token.item() == tokenizer.eos_token_id or 
                next_token.item() == tokenizer.pad_token_id):
                break
            
            # Check for im_end token
            try:
                token_text = tokenizer.decode([next_token.item()])
                if "<|im_end|>" in token_text or "</s>" in token_text:
                    break
            except:
                pass
            
            # Add token
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=-1)
            
            # ✅ Stop nếu quá nhiều repeated tokens
            if step > 20:
                last_tokens = generated_ids[0, -10:].tolist()
                if len(set(last_tokens)) <= 3:  # Quá nhiều repetition
                    print("⚠️ Detected repetition, stopping")
                    break
    
    # Decode response
    try:
        full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        print(f"🔍 Full generated text: {full_text}")
        
        # Extract chỉ phần assistant response
        if "<|im_start|>assistant\n" in full_text:
            response = full_text.split("<|im_start|>assistant\n")[-1]
            if "<|im_end|>" in response:
                response = response.split("<|im_end|>")[0]
            response = response.strip()
        else:
            # Fallback: lấy phần sau prompt
            response = full_text[len(prompt):].strip()
        
        if len(response) < 2:
            return "❌ Response quá ngắn"
        
        return response
        
    except Exception as e:
        print(f"❌ Decode error: {e}")
        return "❌ Lỗi decode"

def simple_generate(model, tokenizer, prompt, max_length=50):
    """Simple generation với greedy decoding để test"""
    model.eval()
    
    inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
    input_ids = inputs['input_ids']
    
    with torch.no_grad():
        generated_ids = input_ids.clone()
        
        for _ in range(max_length):
            outputs = model(generated_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            # ✅ Greedy decoding (chọn token có xác suất cao nhất)
            next_token = torch.argmax(logits[0, -1, :], dim=-1, keepdim=True)
            
            # Stop conditions
            if (next_token.item() == tokenizer.eos_token_id or 
                next_token.item() == tokenizer.pad_token_id):
                break
                
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=-1)
            
            # Early stop if repetitive
            if generated_ids.shape[1] > 20:
                last_5 = generated_ids[0, -5:].tolist()
                if len(set(last_5)) <= 2:
                    break
    
    try:
        full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        if "<|im_start|>assistant\n" in full_text:
            response = full_text.split("<|im_start|>assistant\n")[-1].strip()
        else:
            response = full_text[len(prompt):].strip()
        return response if len(response) > 2 else "❌ Empty response"
    except:
        return "❌ Decode error"

def test_base_model(use_original_qwen=False):
    """Test base model trước khi fine-tune"""
    model_type = "ORIGINAL QWEN2" if use_original_qwen else "CUSTOM"
    print(f"🧪 TESTING {model_type} BASE MODEL")
    print("="*60)
    
    # Load base model
    model, tokenizer = load_base_model(use_original_qwen)
    
    # ✅ Test với câu hỏi cơ bản
    test_questions = [
        "Bạn là ai?",
        "Xin chào!",
        "Bạn có thể làm gì?",
        "Hôm nay thế nào?",
        "Cảm ơn bạn"
    ]
    
    print(f"\n🇻🇳 Testing với {len(test_questions)} câu hỏi cơ bản:")
    print("-" * 60)
    
    successful_tests = 0
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n[{i}] ❓ {question}")
        print("🤖 HyperMamba: ", end="", flush=True)
        
        try:
            # ✅ Test với cả 2 method
            print("Advanced sampling: ", end="", flush=True)
            response1 = test_qwen2_format_generation(
                model, tokenizer, question, max_length=80
            )
            print(response1[:100] + "..." if len(response1) > 100 else response1)
            
            print("Greedy decoding: ", end="", flush=True)
            simple_prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
            response2 = simple_generate(model, tokenizer, simple_prompt, max_length=30)
            print(response2)
            
            # ✅ Chọn response tốt hơn
            response = response2 if len(response2) > 5 and not response2.startswith("❌") else response1
            
            # ✅ Basic validation
            if (len(response) > 5 and 
                not response.startswith("❌") and 
                not any(char in response for char in ["�", "�", "▁"]) and  # Check for broken chars
                len([c for c in response if c.isalpha()]) > 3):
                successful_tests += 1
                print("   ✅ Valid response")
            else:
                print("   ❌ Invalid/broken response")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Final results
    success_rate = successful_tests / len(test_questions) * 100
    print(f"\n{'='*60}")
    print(f"📊 BASE MODEL TEST RESULTS:")
    print(f"   Total tests: {len(test_questions)}")
    print(f"   Successful: {successful_tests}")
    print(f"   Success rate: {success_rate:.1f}%")
    
    if success_rate > 50:
        print("✅ Base model hoạt động OK, có thể fine-tune!")
    elif success_rate > 20:
        print("⚠️ Base model có vấn đề nhẹ, nên check tokenizer")
    else:
        print("❌ Base model có vấn đề nghiêm trọng!")
        print("💡 Khuyến nghị:")
        print("   - Check tokenizer compatibility")
        print("   - Check model vocab_size vs tokenizer vocab_size")
        print("   - Test với model khác như Qwen2-0.5B")
    
    print("="*60)

def debug_tokenizer():
    """Debug tokenizer để tìm vấn đề"""
    print("\n🔍 DEBUGGING TOKENIZER")
    print("="*40)
    
    _, tokenizer = load_base_model()
    
    # Test basic tokenization
    test_text = "Xin chào, bạn là ai?"
    print(f"Test text: {test_text}")
    
    # Test encoding
    tokens = tokenizer.encode(test_text, add_special_tokens=True)
    print(f"Tokens: {tokens}")
    print(f"Max token ID: {max(tokens)}")
    print(f"Tokenizer vocab_size: {tokenizer.vocab_size}")
    
    # Test decoding
    decoded = tokenizer.decode(tokens, skip_special_tokens=False)
    print(f"Decoded: {decoded}")
    
    # Test chat template
    try:
        messages = [{"role": "user", "content": test_text}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print(f"Chat template: {formatted}")
    except Exception as e:
        print(f"Chat template error: {e}")

if __name__ == "__main__":
    print("🧪 HyperMamba vs Original Qwen2 Test")
    print("="*50)
    
    choice = input("\nChọn test mode:\n1. Test custom model\n2. Test ORIGINAL Qwen2-0.6B\n3. So sánh cả hai\n4. Debug tokenizer\n\nLựa chọn (1-4): ").strip()
    
    if choice == "1":
        test_base_model(use_original_qwen=False)
    elif choice == "2":
        test_base_model(use_original_qwen=True)
    elif choice == "3":
        print("\n" + "="*60)
        print("📊 SO SÁNH CUSTOM MODEL vs ORIGINAL QWEN2")
        print("="*60)
        
        print("\n🤖 Testing CUSTOM model:")
        print("-" * 40)
        test_base_model(use_original_qwen=False)
        
        print("\n🔥 Testing ORIGINAL Qwen2:")
        print("-" * 40)
        test_base_model(use_original_qwen=True)
    elif choice == "4":
        debug_tokenizer()
    else:
        print("Chạy test custom model...")
        test_base_model(use_original_qwen=False)
