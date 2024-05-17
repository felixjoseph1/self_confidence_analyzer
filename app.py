import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from flask import jsonify 
from analysis import *

app = Flask(__name__,template_folder='template')

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Perform analysis
        video_path = 'uploads/{}'.format(filename)
        audio_path = 'uploads/extracted_audio.wav'
        extract_audio(video_path, audio_path)
        transcribed_text = transcribe_audio(audio_path)
        voice_confidence = measure_voice_confidence(transcribed_text)
        face_confidence = measure_face_confidence(video_path)
        gesture_confidence = detect_gestures(video_path)
        overall_confidence = f"{round(calculate_overall_confidence(face_confidence, voice_confidence,gesture_confidence)*100, 2)}%"
        #print("Overall Confidence:", round(overall_confidence * 100, 2), "%")

        return jsonify({
            'confidence': overall_confidence,
            'face_confidence': face_confidence,
            'voice_confidence': voice_confidence,
            'gesture_confidence': gesture_confidence,
            'transcribed_text': transcribed_text
        })
    else:
        return jsonify({'error': 'File type not allowed'})


if __name__ == '__main__':
    app.run(debug=True)
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
