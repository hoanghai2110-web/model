
import torch
from model import setup_for_conversation_finetuning
from fine_tune import ConversationDataset
import json

def debug_training():
    print("🔧 Debugging Training Setup...")
    
    # 1. Check model and tokenizer
    model, tokenizer, optimizer, scheduler, config = setup_for_conversation_finetuning()
    
    print(f"✅ Model vocab_size: {config.vocab_size}")
    print(f"✅ Tokenizer vocab_size: {tokenizer.vocab_size}")
    print(f"✅ Model embedding size: {model.token_embedding.weight.shape}")
    
    # 2. Check LoRA parameters
    lora_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            lora_params += param.numel()
            print(f"Trainable: {name} - {param.shape}")
    
    print(f"✅ Total params: {total_params:,}")
    print(f"✅ LoRA params: {lora_params:,}")
    print(f"✅ LoRA ratio: {lora_params/total_params*100:.2f}%")
    
    # 3. Test dataset loading
    print("\n📊 Testing Dataset Loading...")
    dataset = ConversationDataset("dataset.jsonl", tokenizer, max_length=256)
    
    if len(dataset) == 0:
        print("❌ Dataset is empty!")
        return
    
    # Test first sample
    sample = dataset[0]
    print(f"✅ Sample shape: {sample['input_ids'].shape}")
    print(f"✅ Sample tokens: {sample['input_ids']}")
    
    # Decode sample
    decoded = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
    print(f"✅ Decoded sample: {decoded[:200]}...")
    
    # 4. Test forward pass
    print("\n🔄 Testing Forward Pass...")
    model.eval()
    
    batch_input_ids = sample['input_ids'].unsqueeze(0)
    batch_attention_mask = sample['attention_mask'].unsqueeze(0)
    batch_labels = sample['labels'].unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        
        print(f"✅ Logits shape: {logits.shape}")
        print(f"✅ Vocab consistency: {logits.shape[-1] == tokenizer.vocab_size}")
        
        # Test loss calculation
        from torch.nn import CrossEntropyLoss
        loss_fn = CrossEntropyLoss(ignore_index=-100)
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch_labels[..., 1:].contiguous()
        
        loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        print(f"✅ Loss value: {loss.item():.4f}")
        
        if loss.item() > 15.0:
            print("⚠️  WARNING: Loss is very high! Check model initialization.")
        elif loss.item() < 0.1:
            print("⚠️  WARNING: Loss is very low! Check label masking.")
        else:
            print("✅ Loss looks reasonable.")
    
    # 5. Test gradient flow
    print("\n📈 Testing Gradient Flow...")
    model.train()
    optimizer.zero_grad()
    
    outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
    
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = batch_labels[..., 1:].contiguous()
    
    loss = loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    
    loss.backward()
    
    # Check gradients
    grad_norm = 0.0
    has_gradient = False
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm += param.grad.data.norm(2).item() ** 2
            has_gradient = True
            if 'lora_' in name:
                print(f"✅ {name} grad norm: {param.grad.data.norm(2).item():.6f}")
    
    grad_norm = grad_norm ** 0.5
    print(f"✅ Total gradient norm: {grad_norm:.6f}")
    
    if not has_gradient:
        print("❌ No gradients found! Check requires_grad settings.")
    elif grad_norm < 1e-8:
        print("⚠️  WARNING: Gradients are too small!")
    elif grad_norm > 100:
        print("⚠️  WARNING: Gradients are too large!")
    else:
        print("✅ Gradient flow looks good.")

if __name__ == "__main__":
    debug_training()
