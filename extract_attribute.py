import os
import json
import subprocess

# Đường dẫn thư mục chứa file mã độc
dataset_benign_folder = './reports/benign'
dataset_ransomware_folder = './reports/ransomware'
output_folder = './reports/'

# Tạo thư mục lưu report nếu chưa có
os.makedirs(output_folder, exist_ok=True)

# Hàm thực thi cuckoorunner.py và lưu report JSON
def run_cuckoo_and_extract(filepath):
    # Neu la ransom
    # output_folder='./reports/ransomware'
    # Neu la benign
    # output_folder='./reports/benign'
    filename = os.path.basename(filepath)
    name_no_ext = os.path.splitext(filename)[0]
    output_file = os.path.join(output_folder, f'report_{name_no_ext}.json')

    try:
        # Chạy script cuckoorunner.py và lấy output
        result = subprocess.run(
            ['python3', 'cuckoorunner.py', filepath],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300  # 5 phút timeout
        )

        if result.returncode != 0:
            print(f"[!] Lỗi khi chạy {filename}: {result.stderr.decode()}")
            return

        output_path=f"./reports/report_{name_no_ext}.json"
        extract_fields(output_path)

    except subprocess.TimeoutExpired:
        print(f"[!] Timeout với {filename}")

# Hàm trích xuất dll, api, mutex
def extract_fields(report_path, check):
    try:
        with open(report_path, 'r') as f:
            data = json.load(f)

        dlls = []
        apis = []
        mutexes = []

        # Trích xuất các trường nếu tồn tại
        if 'behavior' in data:
            if 'summary' in data['behavior']:
                summary = data['behavior']['summary']
                if 'mutex' in summary:
                    mutexes = summary.get('mutex', [])
                
                if 'dll_loaded' in summary:
                    dlls = summary.get('dll_loaded', [])

            if 'processes' in data['behavior']:
                for process in data['behavior']['processes']:
                    if 'calls' in process:
                        for call in process['calls']:
                            if 'api' in call:
                                apis.append(call['api'])
            
        
        attributes = {
            "dlls": dlls[:10],
            "apis": list(set(apis))[:500],  # loại trùng và lấy 500 đầu
            "mutexes": mutexes[:10]
        }
        report_file = os.path.splitext(os.path.basename(report_path))[0]
        os.makedirs('./attributes', exist_ok=True)
        if check == 1:
            os.makedirs('./attributes/ransomware', exist_ok=True)
            attr_output_path = f'./attributes/ransomware/extract_{report_file}.json'
        else:
            os.makedirs('./attributes/benign', exist_ok=True)
            attr_output_path = f'./attributes/benign/extract_{report_file}.json'
        with open(attr_output_path, 'w') as f:
            json.dump(attributes, f, indent=4)

        print(f"[+] Đã ghi thông tin vào: {attr_output_path}")


    except Exception as e:
        print(f"[!] Lỗi đọc hoặc phân tích report {report_path}: {e}")

# Lặp qua từng file .exe và chạy
#for file in os.listdir(dataset_folder):
#    if file.endswith('.exe'):
#        full_path = os.path.join(dataset_folder, file)
#        run_cuckoo_and_extract(full_path)
# Chỗ này mở ra nếu muốn chạy flow lớn

for filename in os.listdir(dataset_benign_folder):
    if filename.endswith('.json'):
        full_path = os.path.join(dataset_benign_folder, filename)
        extract_fields(full_path,0)

for filename in os.listdir(dataset_ransomware_folder):
    if filename.endswith('.json'):
        full_path = os.path.join(dataset_ransomware_folder, filename)
        extract_fields(full_path,1)