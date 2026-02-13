from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64
import numpy as np

app = Flask(__name__)
CORS(app)

def analyze_image_simple(image):
    """Simple AI detection using heuristics"""
    img_array = np.array(image.convert('RGB'))
    width, height = image.size
    
    scores = []
    
    # Check 1: AI-standard dimensions
    ai_sizes = [512, 768, 1024, 1536, 2048]
    if width in ai_sizes and height in ai_sizes:
        scores.append(80)
    elif width == height and width >= 512:
        scores.append(65)
    else:
        scores.append(30)
    
    # Check 2: Color saturation
    hsv = np.array(image.convert('HSV'))
    saturation = hsv[:, :, 1]
    avg_sat = float(np.mean(saturation))
    
    if avg_sat > 170:
        scores.append(85)
    elif avg_sat > 140:
        scores.append(70)
    elif avg_sat > 100:
        scores.append(50)
    else:
        scores.append(25)
    
    # Check 3: Brightness uniformity
    brightness = hsv[:, :, 2]
    brightness_std = float(np.std(brightness))
    
    if brightness_std < 25:
        scores.append(80)
    elif brightness_std < 45:
        scores.append(65)
    elif brightness_std < 60:
        scores.append(45)
    else:
        scores.append(20)
    
    # Check 4: Color variance
    colors = img_array.reshape(-1, 3)
    color_variance = float(np.var(colors, axis=0).mean())
    
    if color_variance < 1500:
        scores.append(75)
    elif color_variance < 3000:
        scores.append(55)
    else:
        scores.append(25)
    
    # Calculate final score
    final_score = float(np.mean(scores))
    return final_score

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        ai_probability = analyze_image_simple(image)
        is_ai = bool(ai_probability > 50)
        
        return jsonify({
            'success': True,
            'is_ai_generated': is_ai,
            'confidence': int(ai_probability),
            'analysis': {
                'width': int(image.width),
                'height': int(image.height),
                'format': str(image.format) if image.format else 'Unknown'
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/health', methods=['GET'])
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'message': 'AI Detector API is running'})

@app.route('/')
def home():
    return jsonify({
        'message': 'AI Image Detector API',
        'status': 'running',
        'endpoints': {
            '/api/health': 'Health check',
            '/api/analyze': 'POST - Analyze image'
        }
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)