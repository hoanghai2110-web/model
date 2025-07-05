
import json
import re
from typing import List, Dict, Any

def is_valid_message(content: str) -> bool:
    """Kiểm tra xem nội dung tin nhắn có hợp lệ không"""
    if not content or not isinstance(content, str):
        return False
    
    # Loại bỏ tin nhắn quá ngắn (dưới 3 ký tự)
    if len(content.strip()) < 3:
        return False
    
    # Loại bỏ tin nhắn chỉ có ký tự đặc biệt hoặc số
    if not re.search(r'[a-zA-ZÀ-ỹ]', content):
        return False
    
    # Loại bỏ tin nhắn có quá nhiều ký tự lặp
    if re.search(r'(.)\1{5,}', content):
        return False
    
    # Loại bỏ tin nhắn có quá nhiều emoji hoặc ký tự đặc biệt
    special_chars = len(re.findall(r'[^\w\s\.,?!àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]', content))
    if special_chars > len(content) * 0.3:
        return False
    
    return True

def is_valid_conversation(data: Dict[str, Any]) -> bool:
    """Kiểm tra xem cuộc hội thoại có hợp lệ không"""
    if not isinstance(data, dict) or 'messages' not in data:
        return False
    
    messages = data['messages']
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    
    # Kiểm tra cấu trúc user-assistant
    user_msg = None
    assistant_msg = None
    
    for msg in messages:
        if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
            return False
        
        if msg['role'] == 'user':
            user_msg = msg['content']
        elif msg['role'] == 'assistant':
            assistant_msg = msg['content']
    
    # Phải có cả user và assistant message
    if not user_msg or not assistant_msg:
        return False
    
    # Kiểm tra nội dung tin nhắn
    if not is_valid_message(user_msg) or not is_valid_message(assistant_msg):
        return False
    
    # Loại bỏ cuộc hội thoại có câu trả lời quá giống nhau
    if user_msg.lower().strip() == assistant_msg.lower().strip():
        return False
    
    return True

def has_inappropriate_content(content: str) -> bool:
    """Kiểm tra nội dung không phù hợp"""
    inappropriate_keywords = [
        'hack', 'crack', 'pirate', 'torrent', 'cheat',
        'drug', 'kill', 'suicide', 'bomb', 'weapon',
        'sex', 'porn', 'nude', 'nsfw',
        'scam', 'fraud', 'steal', 'rob',
    ]
    
    content_lower = content.lower()
    return any(keyword in content_lower for keyword in inappropriate_keywords)

def clean_dataset(input_file: str = "dataset.jsonl", output_file: str = "dataset_cleaned.jsonl"):
    """Làm sạch dataset và tạo file mới"""
    
    print("🧹 Bắt đầu làm sạch dataset...")
    
    valid_conversations = []
    total_lines = 0
    errors = []
    
    # Đọc và xử lý từng dòng
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                total_lines += 1
                line = line.strip()
                
                if not line:
                    continue
                
                try:
                    # Parse JSON
                    data = json.loads(line)
                    
                    # Kiểm tra tính hợp lệ
                    if is_valid_conversation(data):
                        # Kiểm tra nội dung không phù hợp
                        user_content = ""
                        assistant_content = ""
                        
                        for msg in data['messages']:
                            if msg['role'] == 'user':
                                user_content = msg['content']
                            elif msg['role'] == 'assistant':
                                assistant_content = msg['content']
                        
                        if (not has_inappropriate_content(user_content) and 
                            not has_inappropriate_content(assistant_content)):
                            valid_conversations.append(data)
                        else:
                            errors.append(f"Dòng {line_num}: Nội dung không phù hợp")
                    else:
                        errors.append(f"Dòng {line_num}: Cấu trúc hội thoại không hợp lệ")
                        
                except json.JSONDecodeError as e:
                    errors.append(f"Dòng {line_num}: Lỗi JSON - {str(e)}")
                except Exception as e:
                    errors.append(f"Dòng {line_num}: Lỗi khác - {str(e)}")
    
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file {input_file}")
        return
    
    # Ghi file mới
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for conversation in valid_conversations:
                f.write(json.dumps(conversation, ensure_ascii=False) + '\n')
        
        print(f"✅ Đã làm sạch dataset:")
        print(f"   📊 Tổng số dòng: {total_lines}")
        print(f"   ✅ Dòng hợp lệ: {len(valid_conversations)}")
        print(f"   ❌ Dòng lỗi: {len(errors)}")
        print(f"   📈 Tỷ lệ thành công: {len(valid_conversations)/total_lines*100:.1f}%")
        print(f"   💾 File sạch: {output_file}")
        
        # Hiển thị một số lỗi
        if errors:
            print(f"\n⚠️  Một số lỗi phát hiện:")
            for error in errors[:10]:  # Chỉ hiển thị 10 lỗi đầu
                print(f"   {error}")
            if len(errors) > 10:
                print(f"   ... và {len(errors) - 10} lỗi khác")
                
    except Exception as e:
        print(f"❌ Lỗi khi ghi file: {e}")

def analyze_dataset(file_path: str = "dataset.jsonl"):
    """Phân tích dataset để tìm vấn đề"""
    
    print("🔍 Phân tích dataset...")
    
    issues = {
        'json_errors': [],
        'missing_fields': [],
        'short_messages': [],
        'special_chars': [],
        'duplicates': [],
        'inappropriate': []
    }
    
    seen_conversations = set()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # Kiểm tra cấu trúc
                    if 'messages' not in data:
                        issues['missing_fields'].append(line_num)
                        continue
                    
                    user_content = ""
                    assistant_content = ""
                    
                    for msg in data['messages']:
                        if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                            issues['missing_fields'].append(line_num)
                            continue
                        
                        if msg['role'] == 'user':
                            user_content = msg['content']
                        elif msg['role'] == 'assistant':
                            assistant_content = msg['content']
                    
                    # Kiểm tra tin nhắn ngắn
                    if len(user_content.strip()) < 3 or len(assistant_content.strip()) < 3:
                        issues['short_messages'].append(line_num)
                    
                    # Kiểm tra ký tự đặc biệt
                    if (not re.search(r'[a-zA-ZÀ-ỹ]', user_content) or 
                        not re.search(r'[a-zA-ZÀ-ỹ]', assistant_content)):
                        issues['special_chars'].append(line_num)
                    
                    # Kiểm tra trùng lặp
                    conversation_key = f"{user_content.strip()[:50]}|{assistant_content.strip()[:50]}"
                    if conversation_key in seen_conversations:
                        issues['duplicates'].append(line_num)
                    else:
                        seen_conversations.add(conversation_key)
                    
                    # Kiểm tra nội dung không phù hợp
                    if (has_inappropriate_content(user_content) or 
                        has_inappropriate_content(assistant_content)):
                        issues['inappropriate'].append(line_num)
                        
                except json.JSONDecodeError:
                    issues['json_errors'].append(line_num)
                except Exception:
                    issues['json_errors'].append(line_num)
    
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file {file_path}")
        return
    
    # Báo cáo kết quả
    print(f"\n📋 BÁO CÁO PHÂN TÍCH:")
    total_issues = sum(len(issue_list) for issue_list in issues.values())
    
    if total_issues == 0:
        print("✅ Dataset không có vấn đề gì!")
    else:
        for issue_type, line_numbers in issues.items():
            if line_numbers:
                print(f"   {issue_type}: {len(line_numbers)} dòng")
                if len(line_numbers) <= 5:
                    print(f"      Dòng: {line_numbers}")
                else:
                    print(f"      Dòng: {line_numbers[:5]}... (+{len(line_numbers)-5} dòng khác)")

if __name__ == "__main__":
    print("🛠️  TOOL LỌC DATASET")
    print("=" * 40)
    
    # Phân tích trước
    analyze_dataset("dataset.jsonl")
    
    print("\n" + "=" * 40)
    
    # Làm sạch dataset
    clean_dataset("dataset.jsonl", "dataset_cleaned.jsonl")
    
    print("\n✨ Hoàn thành! Sử dụng file 'dataset_cleaned.jsonl' để training.")
