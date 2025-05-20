import subprocess

# Clear Attributes trước
# Các report để trong reports/benign nếu là lành tính hoặc reports/ransomware nếu là ransomeware
scripts = [
    "extract_attribute.py",
    "filter_reports.py",
    "train.py",
    "train_ransome_mal.py",
    "explain.py",
    "explain_ransome_mal.py"
]

for script in scripts:
    print(f"\n🟢 Running {script} ...")
    subprocess.run(["python", script], check=True)
