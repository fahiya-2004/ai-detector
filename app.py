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
    
    print(f"\n{'='*50}")
    print(f"Analyzing: {width}x{height}")
    print(f"{'='*50}")
    
    # Check 1: AI-standard dimensions
    ai_sizes = [512, 768, 1024, 1536, 2048]
    if width in ai_sizes and height in ai_sizes:
        scores.append(80)
        print(f"✗ Exact AI size: {width}x{height}")
    elif width == height and width >= 512:
        scores.append(65)
        print(f"⚠️  Square image: {width}x{height}")
    else:
        scores.append(30)
        print(f"✓ Non-standard size: {width}x{height}")
    
    # Check 2: Color saturation
    hsv = np.array(image.convert('HSV'))
    saturation = hsv[:, :, 1]
    avg_sat = float(np.mean(saturation))
    
    if avg_sat > 170:
        scores.append(85)
        print(f"✗ Very high saturation: {avg_sat:.1f}")
    elif avg_sat > 140:
        scores.append(70)
        print(f"⚠️  High saturation: {avg_sat:.1f}")
    elif avg_sat > 100:
        scores.append(50)
        print(f"→ Moderate saturation: {avg_sat:.1f}")
    else:
        scores.append(25)
        print(f"✓ Low saturation: {avg_sat:.1f}")
    
    # Check 3: Brightness uniformity
    brightness = hsv[:, :, 2]
    brightness_std = float(np.std(brightness))
    
    if brightness_std < 25:
        scores.append(80)
        print(f"✗ Perfect lighting: {brightness_std:.1f}")
    elif brightness_std < 45:
        scores.append(65)
        print(f"⚠️  Uniform lighting: {brightness_std:.1f}")
    elif brightness_std < 60:
        scores.append(45)
        print(f"→ Moderate lighting: {brightness_std:.1f}")
    else:
        scores.append(20)
        print(f"✓ Natural lighting: {brightness_std:.1f}")
    
    # Check 4: Color variance
    colors = img_array.reshape(-1, 3)
    color_variance = float(np.var(colors, axis=0).mean())
    
    if color_variance < 1500:
        scores.append(75)
        print(f"✗ Low color variance: {color_variance:.1f}")
    elif color_variance < 3000:
        scores.append(55)
        print(f"⚠️  Medium variance: {color_variance:.1f}")
    else:
        scores.append(25)
        print(f"✓ High color variance: {color_variance:.1f}")
    
    # Check 5: Edge smoothness
    gray = img_array.mean(axis=2)
    edges_h = np.abs(np.diff(gray, axis=0))
    edges_v = np.abs(np.diff(gray, axis=1))
    edge_mean = float((edges_h.mean() + edges_v.mean()) / 2)
    
    if edge_mean < 5:
        scores.append(85)
        print(f"✗ Too smooth edges: {edge_mean:.2f}")
    elif edge_mean < 12:
        scores.append(65)
        print(f"⚠️  Smooth edges: {edge_mean:.2f}")
    elif edge_mean < 20:
        scores.append(45)
        print(f"→ Moderate edges: {edge_mean:.2f}")
    else:
        scores.append(25)
        print(f"✓ Natural edges: {edge_mean:.2f}")
    
    # Calculate final score
    final_score = float(np.mean(scores))
    
    print(f"\n{'='*50}")
    print(f"Individual scores: {scores}")
    print(f"Average: {final_score:.1f}%")
    print(f"Result: {'AI GENERATED' if final_score > 50 else 'REAL IMAGE'}")
    print(f"{'='*50}\n")
    
    return final_score

@app.route('/analyze', methods=['POST'])
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
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

@app.route('/')
def home():
    return jsonify({
        'message': 'AI Image Detector API',
        'status': 'running',
        'endpoints': {
            '/health': 'Health check',
            '/analyze': 'POST - Analyze image'
        }
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    print(f"\n🚀 Starting AI Image Detector on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)