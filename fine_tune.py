import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
import os
import math
from model import setup_for_conversation_finetuning

class ConversationDataset(Dataset):
    """Dataset cho format messages trong dataset.jsonl"""

    def __init__(self, jsonl_file, tokenizer, max_length=384):
        self.texts = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load dữ liệu từ dataset.jsonl
        valid_count = 0
        total_count = 0

        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    total_count += 1
                    try:
                        data = json.loads(line)

                        # Xử lý format messages như trong dataset.jsonl
                        if 'messages' in data and len(data['messages']) >= 2:
                            user_msg = None
                            assistant_msg = None

                            for msg in data['messages']:
                                if msg['role'] == 'user':
                                    user_msg = msg['content'].strip()
                                elif msg['role'] == 'assistant':
                                    assistant_msg = msg['content'].strip()

                            if (user_msg and assistant_msg and 
                                len(user_msg) > 3 and len(assistant_msg) > 3):

                                # ✅ SỬ DỤNG CHAT TEMPLATE để format đúng
                                messages = [
                                    {"role": "user", "content": user_msg},
                                    {"role": "assistant", "content": assistant_msg}
                                ]

                                try:
                                    # Dùng chat template của tokenizer
                                    text = self.tokenizer.apply_chat_template(
                                        messages, 
                                        tokenize=False, 
                                        add_generation_prompt=False
                                    )

                                    # ✅ CRITICAL: Test tokenization immediately - use actual vocab size
                                    test_tokens = self.tokenizer.encode(text, add_special_tokens=False)
                                    actual_vocab_size = len(self.tokenizer.get_vocab())
                                    max_valid_token_id = max(actual_vocab_size, 
                                                           getattr(self.tokenizer, 'eos_token_id', 0),
                                                           getattr(self.tokenizer, 'pad_token_id', 0)) + 100

                                    if any(token_id >= max_valid_token_id for token_id in test_tokens):
                                        print(f"⚠️  Skipping sample with invalid tokens: max={max(test_tokens)} >= {max_valid_token_id}")
                                        continue

                                except Exception as e:
                                    print(f"⚠️  Chat template failed: {e}, using manual format")
                                    # Fallback format manual - KHÔNG sử dụng vì có thể tạo bad tokens
                                    continue

                                self.texts.append(text)
                                valid_count += 1
                        else:
                            print(f"⚠️  Dòng {total_count}: Không có messages hoặc format không đúng")

                    except Exception as e:
                        print(f"❌ Lỗi dòng {total_count}: {e}")
                        continue

        print(f"📊 Dataset loaded: {valid_count}/{total_count} valid texts")
        if self.texts:
            print(f"📝 First example: {self.texts[0][:100]}...")
            print(f"📝 Last example: {self.texts[-1][:100]}...")
        else:
            print("❌ Không có dữ liệu hợp lệ!")

        if valid_count < 10:
            print("⚠️  Warning: Dataset quá nhỏ, có thể cần thêm dữ liệu")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Validation: text phải có format Qwen2
        if '<|im_start|>' not in text or '<|im_end|>' not in text:
            print(f"⚠️  Warning: Text không đúng format Qwen2: {text[:50]}...")
            return None

        # ✅ Tokenize ĐÚNG với Qwen2 format
        try:
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt',
                add_special_tokens=False  # ✅ QUAN TRỌNG: Đã có special tokens trong text rồi
            )

            input_ids = encoding['input_ids'].squeeze()
            attention_mask = encoding['attention_mask'].squeeze()

            # ✅ CRITICAL: Validate token IDs with proper buffer
            max_token_id = input_ids.max().item()
            # Allow for special tokens that might be outside base vocab
            max_allowed_id = max(self.tokenizer.vocab_size, 
                               getattr(self.tokenizer, 'eos_token_id', 0),
                               getattr(self.tokenizer, 'pad_token_id', 0)) + 10

            if max_token_id >= max_allowed_id:
                print(f"❌ REJECTING: Token ID {max_token_id} >= max_allowed {max_allowed_id}")
                print(f"Text: {text[:50]}...")
                return None

            # ✅ Additional validation: Check for any negative or obviously wrong token IDs
            if input_ids.min().item() < 0:
                print(f"❌ REJECTING: Negative token ID found")
                return None

        except Exception as e:
            print(f"❌ Tokenization error: {e}")
            return None

        # Labels = input_ids cho causal LM
        labels = input_ids.clone()

        # Mask padding tokens
        labels[attention_mask == 0] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def validate_qwen2_format():
    """Validate dataset format cho Qwen2 với tokenizer check"""
    print("🔍 Validating Qwen2 format...")

    # Load tokenizer để test
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("./qwen2_tokenizer", local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"✅ Tokenizer loaded: vocab_size = {tokenizer.vocab_size}")

        # Debug: Check special tokens
        print(f"🔍 Special tokens:")
        print(f"   eos_token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
        print(f"   pad_token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
        print(f"   chat_template available: {tokenizer.chat_template is not None}")

    except Exception as e:
        print(f"❌ Cannot load tokenizer: {e}")
        return False

    with open("dataset.jsonl", 'r', encoding='utf-8') as f:
        sample = json.loads(f.readline())

    # Extract sample conversation
    if 'messages' in sample:
        user_msg = None
        assistant_msg = None
        for msg in sample['messages']:
            if msg['role'] == 'user':
                user_msg = msg['content']
            elif msg['role'] == 'assistant':
                assistant_msg = msg['content']

        if user_msg and assistant_msg:
            # ✅ SỬ DỤNG CHAT TEMPLATE thay vì format manual
            try:
                messages = [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg}
                ]

                # Use proper chat template
                correct_format = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=False
                )

                print("✅ Correct Qwen2 format (from chat_template):")
                print(correct_format[:100] + "...")

                # ✅ CRITICAL: Test tokenization with proper settings
                # Get actual vocab size including all special tokens
                actual_vocab_size = len(tokenizer.get_vocab())

                tokens = tokenizer.encode(
                    correct_format, 
                    add_special_tokens=False,  # Template already has special tokens
                    truncation=False
                )

                max_token = max(tokens) if tokens else 0

                # ✅ STRICT VALIDATION with actual vocab size
                if max_token >= actual_vocab_size:
                    print(f"❌ CRITICAL ERROR: Token {max_token} >= actual_vocab_size {actual_vocab_size}")
                    print(f"Tokenizer vocab_size: {tokenizer.vocab_size}")
                    print(f"Problematic text: {correct_format}")

                    # Debug: Find problematic tokens
                    for i, token_id in enumerate(tokens):
                        if token_id >= actual_vocab_size:
                            print(f"   Bad token at position {i}: {token_id}")
                            try:
                                token_text = tokenizer.decode([token_id])
                                print(f"   Token text: '{token_text}'")
                            except:
                                print(f"   Cannot decode token {token_id}")
                            break
                    return False

                # Round-trip test
                decoded = tokenizer.decode(tokens, skip_special_tokens=False)
                if decoded != correct_format:
                    print(f"❌ TOKENIZATION MISMATCH:")
                    print(f"Original:  {repr(correct_format[:50])}")
                    print(f"Decoded:   {repr(decoded[:50])}")
                    return False

                print(f"✅ Tokenization test PASSED (max_token: {max_token})")
                return True

            except Exception as e:
                print(f"❌ Chat template error: {e}")
                print(f"Trying manual format...")

                # Fallback to manual format if chat_template fails
                correct_format = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>"

                try:
                    tokens = tokenizer.encode(correct_format, add_special_tokens=False)
                    max_token = max(tokens) if tokens else 0

                    if max_token >= tokenizer.vocab_size:
                        print(f"❌ CRITICAL ERROR: Token {max_token} >= vocab_size {tokenizer.vocab_size}")
                        return False

                    print(f"✅ Manual format works (max_token: {max_token})")
                    return True

                except Exception as e2:
                    print(f"❌ Manual format also failed: {e2}")
                    return False

    print("❌ Invalid format detected!")
    return False

def validate_batch(model, tokenizer, batch, device):
    """Enhanced validation để detect actual learning"""
    model.eval()

    with torch.no_grad():
        input_ids = batch['input_ids'][:1].to(device)

        # ✅ TEST: Forward pass và check multiple tokens
        outputs = model(input_ids)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs

        # Get top 5 predictions for last token
        last_token_logits = logits[0, -1, :]
        top5_tokens = torch.topk(last_token_logits, 5)

        # Decode và check quality
        predictions = []
        for token_id in top5_tokens.indices:
            try:
                token = tokenizer.decode([token_id.item()])
                if token and not token.startswith('[UNK]') and len(token.strip()) > 0:
                    predictions.append(token)
            except:
                continue

        # ✅ BETTER VALIDATION: Check if model predicts reasonable Vietnamese tokens
        vietnamese_indicators = ['a', 'e', 'i', 'o', 'u', 'ă', 'â', 'ê', 'ô', 'ơ', 'ư', 
                               'á', 'à', 'ả', 'ã', 'ạ', ' ', 'nh', 'ng', 'ch', 'th', 'tr']

        valid_predictions = 0
        for pred in predictions[:3]:  # Check top 3
            if any(indicator in pred.lower() for indicator in vietnamese_indicators):
                valid_predictions += 1

        return valid_predictions > 0

def improved_fine_tune():
    """Fine-tuning cải tiến với validation và learning strategy tốt hơn"""

    print("🚀 Starting IMPROVED Fine-tuning")
    print("="*60)

    # ✅ Validate format trước khi bắt đầu
    if not validate_qwen2_format():
        print("❌ Dataset format không đúng cho Qwen2!")
        return

    # Setup model
    model, tokenizer, optimizer, scheduler, config = setup_for_conversation_finetuning()

    # Sửa tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load dataset với format messages
    dataset = ConversationDataset("dataset.jsonl", tokenizer, max_length=384)

    if len(dataset) < 20:
        print("❌ Dataset quá nhỏ để fine-tune hiệu quả!")
        return

    # ✅ Custom collate function để filter None samples
    def collate_fn(batch):
        # Filter out None samples
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None

        # Stack tensors
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

    # DataLoader với custom collate function
    dataloader = DataLoader(
        dataset, 
        batch_size=4,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_fn
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # ✅ Optimizer configuration cải tiến
    # Chỉ optimize LoRA parameters với learning rate cao hơn
    lora_params = [p for name, p in model.named_parameters() if 'lora_' in name and p.requires_grad]

    # Debug: Print actual LoRA parameters
    print(f"🔍 Debug LoRA parameters:")
    total_lora_params = 0
    for name, param in model.named_parameters():
        if 'lora_' in name and param.requires_grad:
            print(f"   {name}: {param.numel():,} params")
            total_lora_params += param.numel()
    print(f"   Total LoRA params: {total_lora_params:,}")

    # ✅ Balanced learning rate for stable training
    optimizer = torch.optim.AdamW(
        lora_params, 
        lr=2e-4,  # ✅ Moderate LR for stable learning
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    print(f"🎯 Optimizer settings:")
    print(f"   Learning rate: 5e-4")
    print(f"   Trainable params: {total_lora_params:,}")
    print(f"   Weight decay: 0.01")

    # ✅ MORE TRAINING - model needs more time to learn
    num_epochs = 10  # Tăng epochs để học kỹ hơn
    total_steps = len(dataloader) * num_epochs
    warmup_steps = total_steps // 20  # 5% warmup

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss function với label smoothing
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)

    print(f"📊 Training setup:")
    print(f"   Dataset size: {len(dataset)}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Total steps: {total_steps}")
    print(f"   Warmup steps: {warmup_steps}")
    print(f"   Device: {device}")
    print(f"   LoRA params: {len(lora_params):,}")

    # Training loop cải tiến
    model.train()
    global_step = 0
    best_loss = float('inf')
    patience = 0
    max_patience = 3  # Early stopping patience

    loss_history = []
    validation_fails = 0

    try:
        for epoch in range(num_epochs):
            print(f"\n📚 Epoch {epoch + 1}/{num_epochs}")
            epoch_loss = 0
            valid_predictions = 0
            total_batches = len(dataloader)

            # ✅ NORMAL TRAINING - one pass per epoch
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for batch_idx, batch in enumerate(progress_bar):
                # ✅ Skip None batches
                if batch is None:
                    continue

                # Move to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs

                # ✅ CRITICAL FIX: Proper label shift for causal LM
                # Model predicts next token, so we shift labels left by 1
                shift_logits = logits[..., :-1, :].contiguous()  # Remove last prediction
                shift_labels = labels[..., 1:].contiguous()       # Remove first label (shift left)

                # Only compute loss on valid tokens (not padding/masked)
                loss = loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping với norm cao hơn để model học được
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

                optimizer.step()
                scheduler.step()

                # Update metrics
                epoch_loss += loss.item()
                global_step += 1

                # ✅ Validation check mỗi 10 steps
                if global_step % 10 == 0:
                    is_valid = validate_batch(model, tokenizer, batch, device)
                    if is_valid:
                        valid_predictions += 1
                    else:
                        validation_fails += 1

                    model.train()  # Quay lại training mode

                    # ✅ CRITICAL DEBUG: Check if weights are actually updating
                    if global_step % 50 == 0:
                        weight_norm = 0.0
                        for name, param in model.named_parameters():
                            if 'lora_' in name and param.requires_grad:
                                weight_norm += param.data.norm(2).item() ** 2
                        weight_norm = weight_norm ** 0.5
                        print(f"🔍 Step {global_step}: LoRA weight norm = {weight_norm:.6f}")

                # ✅ ENHANCED DEBUG LOGGING
                if global_step % 5 == 0:  # Log thường xuyên hơn
                    current_lr = optimizer.param_groups[0]['lr']
                    avg_loss = epoch_loss / (batch_idx + 1)
                    perplexity = math.exp(min(avg_loss, 10))

                    # Check gradient norms
                    grad_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            grad_norm += p.grad.data.norm(2).item() ** 2
                    grad_norm = grad_norm ** 0.5

                    # Check if loss is actually changing
                    loss_change = "N/A"
                    if len(loss_history) > 0:
                        loss_change = f"{loss.item() - loss_history[-1]:.6f}"

                    progress_bar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Δ': loss_change,
                        'Avg': f'{avg_loss:.4f}',
                        'PPL': f'{perplexity:.2f}',
                        'LR': f'{current_lr:.8f}',
                        'GradNorm': f'{grad_norm:.4f}',
                        'Valid': f'{valid_predictions}/{validation_fails+valid_predictions}'
                    })

                    # ✅ CRITICAL: Log mỗi loss value để track
                    if global_step % 20 == 0:
                        print(f"\n📊 Step {global_step}: Loss={loss.item():.6f}, LR={current_lr:.8f}, GradNorm={grad_norm:.6f}")

                        # Check nếu gradient quá nhỏ hoặc quá lớn
                        if grad_norm < 1e-6:
                            print("⚠️  WARNING: Gradient norm rất nhỏ - có thể learning rate quá thấp!")
                        elif grad_norm > 10:
                            print("⚠️  WARNING: Gradient norm rất lớn - có thể learning rate quá cao!")

                        # ✅ Check nếu loss đang TĂNG (exploding)
                        if len(loss_history) >= 5:
                            recent_losses = loss_history[-5:]
                            if all(recent_losses[i] < recent_losses[i+1] for i in range(len(recent_losses)-1)):
                                print("🚨 CRITICAL: Loss đang tăng liên tục - STOPPING TRAINING!")
                                print("Có thể learning rate quá cao hoặc data có vấn đề")
                                break

                            loss_std = torch.tensor(recent_losses).std().item()
                            if loss_std < 0.001:
                                print("⚠️  WARNING: Loss đã không thay đổi trong 5 steps - có thể bị stuck!")

                    loss_history.append(loss.item())

        # End of epoch stats
            avg_epoch_loss = epoch_loss / total_batches
            loss_history.append(avg_epoch_loss)

            print(f"✅ Epoch {epoch+1} completed:")
            print(f"   Average loss: {avg_epoch_loss:.4f}")
            print(f"   Valid predictions: {valid_predictions}/{total_batches}")
            print(f"   Validation success rate: {valid_predictions/total_batches*100:.1f}%")

            # ✅ Early stopping improved - chỉ stop khi loss < 5.0
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                patience = 0
                print("💾 New best loss! Saving checkpoint...")
            else:
                patience += 1
                print(f"⏳ No improvement for {patience} epochs")

            # ✅ PROPER EARLY STOPPING - only stop when actually learning
            if best_loss < 3.0 and patience >= 5:
                print(f"🛑 Early stopping - model learned well (loss={best_loss:.4f})")
                break
            elif patience >= 15:  # More patience for learning
                print(f"🛑 Force stopping after {patience} epochs without improvement")
                break

    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted at epoch {epoch + 1}, step {global_step}")
        print("💾 Saving current progress...")
        avg_epoch_loss = epoch_loss / max(batch_idx + 1, 1) if 'batch_idx' in locals() else best_loss
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        print("💾 Saving current progress...")
        avg_epoch_loss = epoch_loss / max(batch_idx + 1, 1) if 'batch_idx' in locals() else best_loss

    # ✅ Save model với metadata đầy đủ
    print("\n💾 Saving improved model...")
    save_dir = "hypermamba_finetuned"
    os.makedirs(save_dir, exist_ok=True)

    # Save LoRA weights
    lora_state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            lora_state[name] = param.detach().cpu().clone()

    torch.save({
        'lora_state_dict': lora_state,
        'config': config,
        'training_info': {
            'epochs_completed': epoch + 1,
            'total_steps': global_step,
            'final_loss': avg_epoch_loss,
            'best_loss': best_loss,
            'dataset_size': len(dataset),
            'validation_success_rate': valid_predictions/max(total_batches, 1),
            'improved_training': True
        },
        'tokenizer_info': {
            'vocab_size': tokenizer.vocab_size,
            'pad_token_id': tokenizer.pad_token_id,
            'eos_token_id': tokenizer.eos_token_id
        }
    }, os.path.join(save_dir, 'lora_weights.pt'))

    print(f"✅ Model saved to {save_dir}/")
    print(f"📊 Final stats:")
    print(f"   Best loss: {best_loss:.4f}")
    print(f"   Validation success: {valid_predictions/max(total_batches, 1)*100:.1f}%")
    print(f"   Training completed in {global_step} steps")

    # ✅ Quick test ngay sau training
    print("\n🧪 QUICK TEST sau khi fine-tune:")
    print("-" * 40)

    model.eval()
    test_questions = ["Bạn là ai?", "Tên bạn là gì?"]

    for question in test_questions:
        print(f"❓ {question}")
        print("🤖 ", end="", flush=True)

        try:
            # ✅ Sử dụng ĐÚNG format cho generation
            prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
            inputs = tokenizer(prompt, return_tensors='pt').to(device)

            with torch.no_grad():
                generated_ids = inputs['input_ids'].clone()
                original_length = generated_ids.shape[1]

                for step in range(50):  # Tăng số bước generation
                    outputs = model(generated_ids)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs

                    # Lấy logits của token cuối cùng với temperature balanced
                    next_token_logits = logits[0, -1, :] / 1.0  # Temperature = 0.7 để cân bằng

                    # Top-p sampling thay vì top-k
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                    # Top-p filtering (nucleus sampling)
                    top_p_threshold = 0.9
                    sorted_indices_to_remove = cumulative_probs > top_p_threshold
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = 0

                    # Áp dụng filtering
                    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('Inf')

                    # Sample next token
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)

                    # Kiểm tra điều kiện dừng
                    if (next_token.item() == tokenizer.eos_token_id or 
                        next_token.item() == tokenizer.pad_token_id or
                        next_token.item() == tokenizer.convert_tokens_to_ids('<|im_end|>') or
                        generated_ids.shape[1] >= 384):  # Fixed max_length
                        break

                    generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=-1)

                # Decode chỉ phần response mới
                full_response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

                if "<|im_start|>assistant\n" in full_response:
                    response = full_response.split("<|im_start|>assistant\n", 1)[1].strip()
                    if "<|im_end|>" in response:
                        response = response.split("<|im_end|>")[0].strip()
                else:
                    # Fallback: decode chỉ phần tokens mới sinh ra
                    new_tokens = generated_ids[0][original_length:]
                    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

                print(response[:100] + ("..." if len(response) > 100 else ""))

        except Exception as e:
            print(f"❌ Lỗi: {e}")

    print("\n✅ IMPROVED FINE-TUNING COMPLETED!")

if __name__ == "__main__":
    improved_fine_tune()