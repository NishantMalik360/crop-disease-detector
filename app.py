import os
import gc
import json
from PIL import Image

# RAM management settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, request, render_template
import numpy as np
import gdown 

app = Flask(__name__)

MODEL_PATH = 'models/crop_disease_model.h5'
FILE_ID = '1U5qeI7-eS3EjC2NrVTdpDR_pUEobjHQQ' # Aapki Drive ID

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
                # ---> JADU YAHAN HAI: Lazy Loading TensorFlow <---
                import tensorflow as tf
                
                # 1. Image loading directly from memory
                img = Image.open(file.stream).convert('RGB')
                img = img.resize((224, 224))
                img_array = tf.keras.utils.img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0).astype('float32')
                
                # 2. Check & Download model
                if not os.path.exists(MODEL_PATH):
                    os.makedirs('models', exist_ok=True)
                    gdown.download(id=FILE_ID, output=MODEL_PATH, quiet=False)

                # 3. Predict
                model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                predictions = model.predict(img_array)
                idx = int(np.argmax(predictions[0]))
                
                # 4. Disease Name Mapping
                disease_name = f"Unknown Disease (Index: {idx})"
                json_path = 'class_indices.json' 
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        class_indices = json.load(f)
                        for key, value in class_indices.items():
                            if int(value) == idx:
                                disease_name = key
                                break
                
                # 5. Cleanup
                del model
                del img_array
                tf.keras.backend.clear_session()
                gc.collect()
                
                return render_template('result.html', prediction=disease_name)
            
            except Exception as e:
                return f"Error Details: {str(e)}"
                
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
