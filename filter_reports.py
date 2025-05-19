import os
import json
import hashlib

# Đường dẫn đến thư mục chứa file JSON
folder_path = './attributes/ransomware'
# Lặp qua tất cả file trong thư mục
for idx, filename in enumerate(os.listdir(folder_path)):
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        print(f"[{idx}] Đang xử lý: {filename}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Kiểm tra xem có đúng định dạng rỗng không
            if data == {"dlls": [], "apis": [], "mutexes": []}:
                os.remove(file_path)
                print(f"Đã xóa: {filename}")

        except (json.JSONDecodeError, OSError) as e:
            print(f"Lỗi khi xử lý {filename}: {e}")

folder_path ="./attributes/malware"
for idx, filename in enumerate(os.listdir(folder_path)):
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        print(f"[{idx}] Đang xử lý: {filename}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Kiểm tra xem có đúng định dạng rỗng không
            if data == {"dlls": [], "apis": [], "mutexes": []}:
                os.remove(file_path)
                print(f"Đã xóa: {filename}")

        except (json.JSONDecodeError, OSError) as e:
            print(f"Lỗi khi xử lý {filename}: {e}")
    
folder_path = "./attributes/malware"
hash_counts = {}
deleted_files = 0
checked_files = 0

for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        checked_files += 1

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()

            # Đếm số lần hash đã xuất hiện
            current_count = hash_counts.get(content_hash, 0)
            if current_count >= 25:
                os.remove(file_path)
                deleted_files += 1
                print(f"🗑️ Đã xóa bản vượt quá giới hạn: {filename}")
            else:
                hash_counts[content_hash] = current_count + 1

        except Exception as e:
            print(f"⚠️ Lỗi khi xử lý {filename}: {e}")

print(f"\n✅ Đã kiểm tra {checked_files} file, xóa {deleted_files} file vượt quá bản sao cho phép.")