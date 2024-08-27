from flask import Flask, request, send_file, render_template
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import os
from model import build_model, preprocess_input

# Define video processing parameters (adjust as needed)
frames_per_batch = 10

app = Flask(__name__)

# Ensure the uploads directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Load the model (consider loading outside the request for efficiency)
model = build_model(units=512)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_path, fourcc, 5.0, (299, 299))
    frames_batch = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (299, 299))
            frames_batch.append(frame)
            if len(frames_batch) == frames_per_batch:
                # Convert frames to a numpy array and preprocess
                video_data = np.array(frames_batch)
                video_data = preprocess_input(video_data)
                video_data = np.expand_dims(video_data, axis=0)

                # Predict with the model
                predictions, attention_weights = model.predict(video_data)

                # Identify important frames based on threshold
                summary_frames_indices = np.where(predictions > 0.5)[1]
                for idx in summary_frames_indices:
                    output_video.write(frames_batch[idx])

                # Reset the frames batch
                frames_batch = []

    except Exception as e:
        print(f"Error processing video: {e}")
        return False  # Indicate error
    finally:
        # Release resources
        cap.release()
        output_video.release()
        return True  # Indicate success

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize_video():
    if 'video' not in request.files:
        return "No video file uploaded", 400
    video_file = request.files['video']
    filename = secure_filename(video_file.filename)
    # Define output path (replace with desired location)
    output_path = os.path.join('uploads', f"summary_{filename}")
    video_path = os.path.join('uploads', filename)
    video_file.save(video_path)

    # Process the video and generate summary
    success = process_video(video_path, output_path)
    if success:
        return render_template('index.html', summarized_video_path=f"summary_{filename}")
    else:
        return render_template('index.html', message="Error processing video")

@app.route('/download/<filename>')
def download_video(filename):
    return send_file(os.path.join('uploads', filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
