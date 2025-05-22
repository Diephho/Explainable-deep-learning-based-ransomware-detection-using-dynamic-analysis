from flask import Flask, request, jsonify, send_file
import os
import uuid
import Check_single as check_ransom_benign
import Check_single_ransome_mal as check_ransom_mal
import traceback

UPLOAD_FOLDER = 'analyze'
REPORT_FOLDER = 'checkfile/reports'
PLOT_FOLDER = 'checkfile/plots'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files.get('file')
    analysis_type = request.form.get('analysis_type')
    input_type = request.form.get('input_type')

    if not file or not analysis_type or not input_type:
        return jsonify({"error": "Thiếu file hoặc tham số"}), 400

    filename = f"{uuid.uuid4()}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    module = None
    if analysis_type == "Ransomware and Benign":
        module = check_ransom_benign
    elif analysis_type == "Ransomware and Malware":
        module = check_ransom_mal
    else:
        return jsonify({"error": "Loại phân tích không hợp lệ"}), 400

    try:
        # Xử lý theo loại input
        if input_type == 'execute':
            task_id = module.submit_sample(filepath)
            module.wait_for_report(task_id)

            report_path = os.path.join(REPORT_FOLDER, f"report_{filename}.json")
            module.download_report(task_id, report_path)

            attr_path = os.path.join(REPORT_FOLDER, f"attributes_{filename}.json")
            module.extract_fields(report_path, attr_path)

            # Hàm này trả về (label, confidence, lime_result) và vẽ plot
            label, confidence, lime_result = module.check_and_explain(attr_path)

        elif input_type == 'report':
            attr_path = os.path.join(REPORT_FOLDER, f"attributes_{filename}.json")
            module.extract_fields(filepath, attr_path)
            label, confidence, lime_result = module.check_and_explain(attr_path)

        elif input_type == 'attribute':
            label, confidence, lime_result = module.check_and_explain(filepath)

        else:
            return jsonify({"error": "Kiểu đầu vào không hợp lệ"}), 400

        # Vẽ biểu đồ LIME rồi lưu ra file plot dưới dạng PNG (giả sử module trả về plot object)
        plot_filename = f"plot_{uuid.uuid4()}.png"
        plot_path = os.path.join(PLOT_FOLDER, plot_filename)
        module.plot_lime_top_5_5_to_file(lime_result, 1, label, confidence, save_path=plot_path)

        # Trả về JSON kết quả + đường dẫn plot để frontend tải
        return jsonify({
            "label": label,
            "confidence": confidence,
            "lime_features": lime_result,
            "plot_url": f"/plot/{plot_filename}"
        })

    except Exception as e:
        print("[ERROR]", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/plot/<filename>')
def serve_plot(filename):
    path = os.path.join(PLOT_FOLDER, filename)
    if os.path.exists(path):
        return send_file(path, mimetype='image/png')
    return jsonify({"error": "File không tồn tại"}), 404

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)

