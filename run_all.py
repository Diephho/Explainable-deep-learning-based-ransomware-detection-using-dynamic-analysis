import subprocess

# Clear Attributes trước
# Các report để trong reports/benign nếu là lành tính hoặc reports/ransomware nếu là ransomeware
scripts = [
    "extract_attribute.py",
    "train.py",
    "explain.py"
]

for script in scripts:
    print(f"\n🟢 Running {script} ...")
    subprocess.run(["python", script], check=True)
