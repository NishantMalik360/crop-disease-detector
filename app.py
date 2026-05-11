import os
# 1. Sabse pehle memory aur legacy settings (Keras 3 ko bypass karne ke liye)
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
from tensorflow.keras.layers import InputLayer
import numpy as np
import json
import gdown
import urllib.parse

# 2. Memory saaf karein
tf.keras.backend.clear_session()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

MODEL_PATH = 'models/crop_disease_model.h5'
JSON_PATH = 'models/class_indices.json'

# --- 3. DISEASE INFO DATA (Ise upar hona chahiye taaki functions ise use kar sakein) ---
DISEASE_INFO = {
    # --- APPLE ---
    "Apple___Apple_scab": {
        "cause": "Caused by the fungus Venturia inaequalis, which thrives in cool, wet spring weather.",
        "fix": "Apply preventative fungicides like captan or myclobutanil. Rake and destroy fallen infected leaves.",
        "tips": "Ensure trees are pruned to allow good air circulation and rapid drying of leaves."
    },
    "Apple___Black_rot": {
        "cause": "Caused by the fungus Botryosphaeria obtusa, infecting dead tissue or wounds.",
        "fix": "Prune out dead or diseased wood. Apply appropriate fungicides during early development.",
        "tips": "Remove mummified fruit from the tree and clean up debris around the base."
    },
    "Apple___Cedar_apple_rust": {
        "cause": "A fungal disease caused by Gymnosporangium juniperi-virginianae that requires juniper plants to complete its lifecycle.",
        "fix": "Use fungicides containing myclobutanil or copper. Remove nearby cedar/juniper hosts if possible.",
        "tips": "Plant rust-resistant apple varieties."
    },
    "Apple___healthy": {
        "cause": "N/A", "fix": "N/A", "tips": "Continue regular watering, pruning, and fertilization schedules."
    },

    # --- BLUEBERRY ---
    "Blueberry___healthy": {
        "cause": "N/A", "fix": "N/A", "tips": "Maintain acidic soil (pH 4.5-5.5) and ensure consistent moisture."
    },

    # --- CHERRY ---
    "Cherry_(including_sour)___Powdery_mildew": {
        "cause": "Caused by the fungus Podosphaera clandestina, favored by high humidity and dry leaves.",
        "fix": "Apply sulfur or potassium bicarbonate sprays. Prune for better airflow.",
        "tips": "Water at the base of the tree to keep leaves dry."
    },
    "Cherry_(including_sour)___healthy": {
        "cause": "N/A", "fix": "N/A", "tips": "Prune annually during dormant season and apply balanced fertilizer."
    },

    # --- CORN (MAIZE) ---
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "cause": "Caused by the fungus Cercospora zeae-maydis, surviving in crop residue.",
        "fix": "Apply foliar fungicides if lesions reach the ear leaf. Practice crop rotation.",
        "tips": "Till the soil to bury infected crop residue from the previous year."
    },
    "Corn_(maize)___Common_rust_": {
        "cause": "Caused by the fungus Puccinia sorghi, favored by cool, moist weather.",
        "fix": "Usually doesn't require treatment, but severe cases can be treated with fungicides.",
        "tips": "Plant rust-resistant corn hybrids."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "cause": "Caused by Exserohilum turcicum fungus, thriving in moderate temperatures and heavy dew.",
        "fix": "Use preventative fungicides before tasseling. Rotate crops away from corn.",
        "tips": "Choose resistant hybrids and manage field residue."
    },
    "Corn_(maize)___healthy": {
        "cause": "N/A", "fix": "N/A", "tips": "Ensure adequate nitrogen levels and deep watering during dry spells."
    },

    # --- GRAPE ---
    "Grape___Black_rot": {
        "cause": "Caused by the fungus Guignardia bidwellii, common in warm, humid climates.",
        "fix": "Apply protective fungicides from early bloom until berries are pea-sized.",
        "tips": "Prune vines properly and destroy infected fruit (mummies)."
    },
    "Grape___Esca_(Black_Measles)": {
        "cause": "A complex of fungi attacking the wood, often infecting through pruning wounds.",
        "fix": "No chemical cure. Remove and burn severely infected vines or trunks.",
        "tips": "Avoid pruning during wet weather to prevent fungal spores from entering wounds."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "cause": "Caused by the fungus Pseudocercospora vitis.",
        "fix": "Use copper-based fungicides. Remove fallen leaves in winter.",
        "tips": "Improve canopy airflow by selective leaf pulling."
    },
    "Grape___healthy": {
        "cause": "N/A", "fix": "N/A", "tips": "Practice good canopy management and winter pruning."
    },

    # --- ORANGE ---
    "Orange___Haunglongbing_(Citrus_greening)": {
        "cause": "Caused by a bacterium spread by the Asian citrus psyllid insect.",
        "fix": "No cure exists. Infected trees must be removed and destroyed to prevent spreading.",
        "tips": "Control psyllid populations with insecticides and plant certified disease-free trees."
    },

    # --- PEACH ---
    "Peach___Bacterial_spot": {
        "cause": "Caused by the bacterium Xanthomonas campestris, favored by warm, wet, and windy weather.",
        "fix": "Apply copper sprays during dormancy and early spring. Avoid excessive nitrogen.",
        "tips": "Plant resistant peach varieties."
    },
    "Peach___healthy": {
        "cause": "N/A", "fix": "N/A", "tips": "Thin fruit early in the season for better sizing and tree health."
    },

    # --- PEPPER (BELL) ---
    "Pepper,_bell___Bacterial_spot": {
        "cause": "Caused by Xanthomonas bacteria, spreading rapidly in warm, rainy weather.",
        "fix": "Spray copper-based bactericides. Remove infected plant debris.",
        "tips": "Water at the base of the plants to avoid splashing bacteria onto leaves."
    },
    "Pepper,_bell___healthy": {
        "cause": "N/A", "fix": "N/A", "tips": "Maintain consistent soil moisture to prevent blossom end rot."
    },

    # --- POTATO ---
    "Potato___Early_blight": {
        "cause": "Caused by the fungus Alternaria solani, often starting on older, lower leaves.",
        "fix": "Apply chlorothalonil or copper fungicides. Remove infected leaves.",
        "tips": "Practice a 2-3 year crop rotation with non-nightshade plants."
    },
    "Potato___Late_blight": {
        "cause": "Caused by the water mold Phytophthora infestans (the Irish Potato Famine pathogen).",
        "fix": "Apply strict preventative fungicide programs. Destroy infected plants immediately.",
        "tips": "Plant certified disease-free seed potatoes and ensure good soil drainage."
    },
    "Potato___healthy": {
        "cause": "N/A", "fix": "N/A", "tips": "Hill soil around the base of the plant to protect developing tubers."
    },

    # --- RASPBERRY ---
    "Raspberry___healthy": {
        "cause": "N/A", "fix": "N/A", "tips": "Prune out old, dead canes after harvest to promote new growth."
    },

    # --- SOYBEAN ---
    "Soybean___healthy": {
        "cause": "N/A", "fix": "N/A", "tips": "Monitor for aphids and ensure proper soil inoculation for nitrogen fixation."
    },

    # --- SQUASH ---
    "Squash___Powdery_mildew": {
        "cause": "A fungal disease thriving in high humidity and shaded conditions.",
        "fix": "Apply neem oil, sulfur, or potassium bicarbonate. Remove severely infected leaves.",
        "tips": "Space plants generously for sunlight and airflow."
    },

    # --- STRAWBERRY ---
    "Strawberry___Leaf_scorch": {
        "cause": "Caused by the fungus Diplocarpon earlianum.",
        "fix": "Remove infected leaves. Apply protective fungicides before flowering.",
        "tips": "Keep the strawberry patch weeded and avoid overhead sprinkling."
    },
    "Strawberry___healthy": {
        "cause": "N/A", "fix": "N/A", "tips": "Renew the strawberry patch every 3-4 years for best production."
    },

    # --- TOMATO ---
    "Tomato___Bacterial_spot": {
        "cause": "Caused by Xanthomonas bacteria spreading via water splash.",
        "fix": "Use copper fungicides. Discard severely infected plants (do not compost).",
        "tips": "Use mulch to prevent soil from splashing onto lower leaves."
    },
    "Tomato___Early_blight": {
        "cause": "Caused by Alternaria linariae, forming target-like spots on lower leaves.",
        "fix": "Apply fungicides like chlorothalonil. Remove lower branches as the plant grows.",
        "tips": "Stake or cage tomatoes to keep foliage off the ground."
    },
    "Tomato___Late_blight": {
        "cause": "Caused by Phytophthora infestans, rapidly destroying leaves and fruit.",
        "fix": "Difficult to cure once started. Use preventative copper sprays. Destroy sick plants.",
        "tips": "Avoid planting tomatoes near potatoes, as they share this disease."
    },
    "Tomato___Leaf_Mold": {
        "cause": "Caused by the fungus Passalora fulva, primarily an issue in humid greenhouses.",
        "fix": "Increase ventilation and reduce humidity. Apply appropriate fungicides.",
        "tips": "Water in the morning so plants dry quickly during the day."
    },
    "Tomato___Septoria_leaf_spot": {
        "cause": "Caused by the fungus Septoria lycopersici, creating small circular spots with dark borders.",
        "fix": "Remove infected leaves immediately. Spray with chlorothalonil or copper.",
        "tips": "Clean up all garden debris at the end of the season to prevent overwintering."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "cause": "Tiny arachnids that suck sap from leaves, thriving in hot, dry conditions.",
        "fix": "Spray leaves forcefully with water. Use insecticidal soap or neem oil.",
        "tips": "Keep plants well-watered, as stressed plants attract mites."
    },
    "Tomato___Target_Spot": {
        "cause": "Caused by the fungus Corynespora cassiicola.",
        "fix": "Apply fungicides. Remove heavily infected lower foliage.",
        "tips": "Ensure good airflow and avoid working in the garden when plants are wet."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "cause": "A virus transmitted primarily by the whitefly insect.",
        "fix": "No cure for the virus. Remove infected plants. Control whitefly populations.",
        "tips": "Use reflective mulches to deter whiteflies and plant resistant varieties."
    },
    "Tomato___Tomato_mosaic_virus": {
        "cause": "A highly contagious virus transmitted by touch, tools, or infected seeds.",
        "fix": "No cure. Destroy infected plants. Wash hands and sterilize tools thoroughly.",
        "tips": "Do not use tobacco products near tomato plants, as the virus can be carried in tobacco."
    },
    "Tomato___healthy": {
        "cause": "N/A", "fix": "N/A", "tips": "Provide consistent moisture to prevent blossom end rot and fruit splitting."
    }
}

# --- MODEL DOWNLOAD ---
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    os.makedirs('models', exist_ok=True) 
    file_id = '1U5qeI7-eS3EjC2NrVTdpDR_pUEobjHQQ'
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, MODEL_PATH, quiet=False)

# --- MODEL LOADING WITH BYPASS ---
try:
    print("Attempting model load...")
    # safe_mode=False aur InputLayer bypass dono use kar rahe hain
    model = tf.keras.models.load_model(
        MODEL_PATH, 
        compile=False, 
        safe_mode=False,
        custom_objects={'InputLayer': InputLayer}
    )
    print("SUCCESS: Model loaded finally!")
except Exception as e:
    print(f"Loading failed: {e}")
    model = None

# --- LOAD CLASS NAMES ---
try:
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        class_indices = json.load(f)
    class_names = {v: k for k, v in class_indices.items()}
except Exception as e:
    print(f"JSON Error: {e}")
    class_names = {}

# --- PREDICTION FUNCTION ---
def predict_image(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    # Memory optimization: explicitly cast to float32
    img_array = np.expand_dims(img_array, axis=0).astype('float32') / 255.0
    
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_index] * 100)
    
    predicted_class = class_names.get(predicted_class_index, "Unknown")
    return predicted_class, confidence

# --- ROUTES ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file and model:
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            disease, confidence = predict_image(filepath)
            
            if confidence < 60.0:
                google_link = f"https://lens.google.com/uploadbyurl?url={request.url_root}{filepath}"
                return render_template('result.html', disease="Unknown / Low Confidence", 
                                     confidence=confidence, img_path=filepath, google_link=google_link)
            
            info = DISEASE_INFO.get(disease, {"cause": "No data", "fix": "No data", "tips": "No data"})
            return render_template('result.html', disease=disease, confidence=confidence, 
                                 info=info, img_path=filepath)
            
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    # threaded=False aur debug=False Render free tier ke liye best hai
    app.run(host='0.0.0.0', port=port, debug=False, threaded=False)
