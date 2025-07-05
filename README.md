
# HyperMamba - Trợ lý AI tiếng Việt

HyperMamba là một mô hình ngôn ngữ nhỏ gọn (130M parameters) được tối ưu cho việc trò chuyện bằng tiếng Việt. Mô hình sử dụng kiến trúc Transformer Lite với LoRA (Low-Rank Adaptation) để fine-tuning hiệu quả.

## 🚀 Tính năng chính

- **Nhẹ và nhanh**: Chỉ 130M parameters, phù hợp chạy trên máy tính cá nhân
- **Tối ưu tiếng Việt**: Sử dụng Qwen2 tokenizer với hỗ trợ tiếng Việt tốt
- **LoRA fine-tuning**: Fine-tune hiệu quả với ít dữ liệu
- **Kiến trúc hiện đại**: RMSNorm, RoPE, SwiGLU, Gated Attention

## 📁 Cấu trúc dự án

```
├── model.py          # Định nghĩa mô hình TransformerLite
├── fine_tune.py      # Script fine-tuning với LoRA
├── test_model.py     # Script test mô hình sau fine-tuning
├── test_tokenizer.py # Test Qwen2 tokenizer
├── dataset.jsonl     # Dữ liệu mẫu cho fine-tuning
├── qwen2_tokenizer/  # Qwen2 tokenizer đã tải về
└── README.md         # File hướng dẫn này
```

## 🛠️ Cài đặt và chạy

### 0. Cài đặt thư viện cần thiết

Trước tiên, cài đặt các thư viện Python cần thiết:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers tokenizers tqdm
```

Hoặc nếu có GPU:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers tokenizers tqdm
```

**Trong Replit**: Các thư viện sẽ được tự động cài đặt khi chạy code lần đầu.

### 1. Fine-tuning mô hình

Trước tiên, bạn cần fine-tune mô hình với dữ liệu tiếng Việt:

```bash
python fine_tune.py
```

Hoặc sử dụng workflow trong Replit:
- Chọn "Fine-tune HyperMamba" từ dropdown workflow

**Quá trình fine-tuning sẽ:**
- Load mô hình base với LoRA configuration
- Đọc dữ liệu từ `dataset.jsonl`
- Fine-tune trong 3 epochs
- Lưu LoRA weights vào folder `hypermamba_finetuned/`

### 2. Test mô hình

Sau khi fine-tuning xong, test mô hình:

```bash
python test_model.py
```

Hoặc sử dụng workflow:
- Chọn "Test HyperMamba" từ dropdown workflow

**Chế độ test:**
1. **Interactive chat**: Trò chuyện tương tác với mô hình
2. **Predefined questions**: Test với câu hỏi có sẵn

## 📊 Dữ liệu training

File `dataset.jsonl` chứa dữ liệu mẫu theo format:

```json
{
  "messages": [
    {"role": "user", "content": "Danh tính của bạn?"},
    {"role": "assistant", "content": "Danh tính của mình là HyperMamba..."}
  ]
}
```

### Thêm dữ liệu mới:
1. Mở file `dataset.jsonl`
2. Thêm dòng mới theo format trên
3. Chạy lại fine-tuning

## ⚙️ Cấu hình mô hình

### Thông số mô hình chính:
- **Vocabulary size**: 151,657 (Qwen2 tokenizer)
- **Hidden size**: 768
- **Layers**: 12
- **Attention heads**: 12
- **Context length**: 4,096 tokens

### Thông số LoRA:
- **Rank**: 16
- **Alpha**: 32  
- **Dropout**: 0.1
- **Target modules**: qkv_proj, out_proj, gate_proj, up_proj, down_proj

## 🎯 Sử dụng

### 1. Chạy interactive chat:

```bash
python test_model.py
# Chọn option 1 khi được hỏi
```

Ví dụ conversation:
```
👤 Bạn: Xin chào!
🤖 HyperMamba: Xin chào! Mình là HyperMamba, rất vui được gặp bạn!

👤 Bạn: Bạn có thể làm gì?
🤖 HyperMamba: Mình có thể trò chuyện với bạn, trả lời câu hỏi, kể chuyện...
```

### 2. Test với câu hỏi có sẵn:

```bash
python test_model.py  
# Chọn option 2 khi được hỏi
```

## 🔧 Tuning và tối ưu

### Để cải thiện chất lượng:

1. **Thêm dữ liệu**: Bổ sung câu hỏi-trả lời vào `dataset.jsonl`
2. **Tăng epochs**: Sửa `num_epochs` trong `fine_tune.py`
3. **Điều chỉnh learning rate**: Thay đổi `lr` trong optimizer
4. **Tăng LoRA rank**: Sửa `lora_rank` trong config (tốn nhiều memory hơn)

### Monitoring training:

Fine-tuning sẽ hiển thị:
- Loss theo từng step
- Average loss theo epoch
- Learning rate hiện tại
- Thông tin lưu model

## 🐛 Troubleshooting

### Lỗi thường gặp:

1. **Import errors (torch, transformers)**:
   ```bash
   pip install torch transformers tokenizers tqdm
   ```
   Hoặc trong Replit: Chờ tự động cài đặt khi chạy code

2. **CUDA out of memory**: 
   - Giảm `batch_size` trong `fine_tune.py`
   - Giảm `max_length` trong dataset

3. **Tokenizer không load được**:
   - Kiểm tra folder `mistral_tokenizer/` có đầy đủ files
   - Model sẽ tự động fallback về remote tokenizer

4. **Response không tốt**:
   - Thêm nhiều dữ liệu training
   - Tăng số epochs
   - Kiểm tra format dữ liệu trong `dataset.jsonl`

5. **ModuleNotFoundError**:
   - Đảm bảo đã cài đặt đủ thư viện
   - Restart kernel nếu cần

## 📈 Performance

### Thông số hiệu suất:
- **Total parameters**: ~130M
- **Trainable LoRA parameters**: ~2M (1.5% của total)
- **Memory usage**: ~2-4GB VRAM (tùy batch size)
- **Training time**: ~5-10 phút (3 epochs trên CPU/GPU)

## 🤝 Đóng góp

Để cải thiện mô hình:
1. Thêm dữ liệu conversation chất lượng cao
2. Tối ưu hyperparameters
3. Thử nghiệm kiến trúc mới

## 📝 Lưu ý

- Mô hình được thiết kế cho mục đích học tập và thử nghiệm
- Chất lượng phụ thuộc vào dữ liệu training
- Khuyến nghị fine-tune với dataset lớn hơn cho production

---

**Happy coding! 🚀**

Nếu có vấn đề gì, hãy kiểm tra console output để debug.
