import os
import gc
import io
import json # <-- JSON import zaroori hai
from PIL import Image

# RAM management settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
import gdown

app = Flask(__name__)

MODEL_PATH = 'models/crop_disease_model.h5'
FILE_ID = '1U5qeI7-eS3EjC2NrVTdpDR_pUEobjHQQ' 
JSON_PATH = 'class_indices.json' # <-- Aapki JSON file ka naam

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        
        file = request.files['file']
        if file.filename == '':
            return "No selected file"

        if file:
            try:
                # 1. Image ko memory se load karein
                img = Image.open(file.stream).convert('RGB')
                img = img.resize((224, 224))
                img_array = tf.keras.utils.img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0).astype('float32')
                
                # 2. Model download check
                if not os.path.exists(MODEL_PATH):
                    os.makedirs('models', exist_ok=True)
                    gdown.download(id=FILE_ID, output=MODEL_PATH, quiet=False)

                # 3. Model load aur predict
                model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                predictions = model.predict(img_array)
                idx = int(np.argmax(predictions[0]))
                
                # --- 4. DISEASES WALA PART ---
                disease_name = "Unknown"
                if os.path.exists(JSON_PATH):
                    with open(JSON_PATH, 'r') as f:
                        class_indices = json.load(f)
                        # JSON usually {"Apple_scab": 0} format mein hota hai
                        # Isko reverse karke {0: "Apple_scab"} banayenge
                        labels = {v: k for k, v in class_indices.items()}
                        disease_name = labels.get(idx, "Unknown Disease")
                        # Underscore hatakar space daalne ke liye
                        disease_name = disease_name.replace('_', ' ')
                else:
                    disease_name = f"Class {idx} (Warning: class_indices.json not found!)"
                # -----------------------------

                # 5. Cleanup (RAM bachaane ke liye)
                del model
                del img_array
                tf.keras.backend.clear_session()
                gc.collect()
                
                # Final Result HTML mein bhejein
                return f"<h2>Analysis Complete!</h2><p>Detected Disease: <b>{disease_name}</b></p><a href='/'>Go Back</a>"
            
            except Exception as e:
                return f"Error Details: {str(e)}"
                
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
