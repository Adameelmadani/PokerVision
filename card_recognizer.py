import cv2
import numpy as np
import tensorflow as tf
import os
import glob

# Map numeric predictions to card ranks and suits
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'J', 'Q', 'K', 'A']  # Removed '10'
SUITS = ['Clubs', 'Diamonds', 'Hearts', 'Spades']

# Load the separate card recognition models if they exist, otherwise use placeholders
models_dir = os.path.join(os.path.dirname(__file__), "models")
rank_model_path = os.path.join(models_dir, "rank_model.h5")
suit_model_path = os.path.join(models_dir, "suit_model.h5")
empty_model_path = os.path.join(models_dir, "empty_model.h5")

# Try to load rank model
try:
    rank_model = tf.keras.models.load_model(rank_model_path)
    rank_model_loaded = True
except (ImportError, FileNotFoundError):
    rank_model_loaded = False
    print(f"Rank model not found at {rank_model_path}. Using placeholder predictions.")

# Try to load suit model
try:
    suit_model = tf.keras.models.load_model(suit_model_path)
    suit_model_loaded = True
except (ImportError, FileNotFoundError):
    suit_model_loaded = False
    print(f"Suit model not found at {suit_model_path}. Using placeholder predictions.")

# Try to load empty position model
try:
    empty_model = tf.keras.models.load_model(empty_model_path)
    empty_model_loaded = True
except (ImportError, FileNotFoundError):
    empty_model_loaded = False
    print(f"Empty position model not found at {empty_model_path}. Using fallback methods.")

def recognize_cards(image, card_regions):
    """
    Recognize cards in the image based on detected regions
    
    Args:
        image (numpy.ndarray): Input image
        card_regions (list): List of card region dictionaries
        
    Returns:
        list: List of detected cards with their ranks and suits
    """
    results = []
    
    for region in card_regions:
        # Extract card image
        card_images = extract_card_image(image, region)
        
        # Preprocess rank/suit image for the model
        rank_suit_img = card_images['rank_suit']
        preprocessed = preprocess_card_image(rank_suit_img)
        
        # Get card rank and suit
        if rank_model_loaded and suit_model_loaded:
            # Use the trained models
            rank, suit = predict_card_with_models(preprocessed)
        else:
            # Placeholder recognition using simple color analysis
            rank, suit = simple_card_recognition(rank_suit_img)
        
        results.append({
            'region': region,
            'rank': rank,
            'suit': suit
        })
        
    return results

def preprocess_card_image(image):
    """Preprocess card image for model input"""
    # Resize to expected input size
    resized = cv2.resize(image, (64, 64))
    # Normalize pixel values
    normalized = resized / 255.0
    return normalized

def predict_card_with_models(preprocessed_image):
    """Predict card rank and suit using the trained models"""
    # Add batch dimension
    img = np.expand_dims(preprocessed_image, axis=0)
    
    # Get predictions from both models
    rank_prediction = rank_model.predict(img, verbose=0)
    suit_prediction = suit_model.predict(img, verbose=0)
    
    # Get the highest probability class for each
    rank_idx = np.argmax(rank_prediction[0])
    suit_idx = np.argmax(suit_prediction[0])
    
    # Ensure indices are within valid range
    rank_idx = min(rank_idx, len(RANKS) - 1)  # Clip to valid index
    suit_idx = min(suit_idx, len(SUITS) - 1)  # Clip to valid index
    
    return RANKS[rank_idx], SUITS[suit_idx]

def simple_card_recognition(image):
    """Simple card recognition as a fallback when no model is available"""
    # This is a placeholder that guesses based on colors
    # In a real implementation, you'd use template matching or more robust methods
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Check if the card has more red (hearts/diamonds) or black (clubs/spades)
    # This is very simplistic and just for demonstration
    average_h = np.mean(hsv[:,:,0])
    average_s = np.mean(hsv[:,:,1])
    
    if average_h < 20 or average_h > 170:
        if average_s > 100:
            suit = 'Hearts'
        else:
            suit = 'Diamonds'
    else:
        if average_s > 50:
            suit = 'Clubs'
        else:
            suit = 'Spades'
    
    # Just return a random rank for demonstration
    import random
    rank = random.choice(RANKS)
    
    return rank, suit

def extract_card_image(image, region):
    """Extract card image from the region"""
    x, y, w, h = region['region']
    card_img = image[y:y+h, x:x+w]
    
    # Extract rank and suit regions (top-left corner of the card)
    rank_height = int(h * 0.2)
    rank_width = int(w * 0.2)
    rank_img = card_img[0:rank_height, 0:rank_width]
    
    return {
        'card': card_img,
        'rank_suit': rank_img
    }

def load_card_templates():
    """
    Load card template images for template matching approach
    
    Returns:
        tuple: (rank_templates, suit_templates) dictionaries with template images
    """
    # Get paths to template directories
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    templates_dir = os.path.join(data_dir, "templates")
    rank_dir = os.path.join(templates_dir, "ranks")
    suit_dir = os.path.join(templates_dir, "suits")
    
    # Create template directories if they don't exist
    os.makedirs(rank_dir, exist_ok=True)
    os.makedirs(suit_dir, exist_ok=True)
    
    # Load rank templates
    rank_templates = {}
    for rank in RANKS:
        rank_path = os.path.join(rank_dir, f"{rank}.png")
        if os.path.exists(rank_path):
            rank_templates[rank] = cv2.imread(rank_path)
        else:
            print(f"Warning: Template for rank {rank} not found at {rank_path}")
            
    # Load suit templates
    suit_templates = {}
    for suit in SUITS:
        suit_path = os.path.join(suit_dir, f"{suit}.png")
        if os.path.exists(suit_path):
            suit_templates[suit] = cv2.imread(suit_path)
        else:
            print(f"Warning: Template for suit {suit} not found at {suit_path}")
    
    if not rank_templates:
        print("No rank templates found. Please add template images to data/templates/ranks/")
    if not suit_templates:
        print("No suit templates found. Please add template images to data/templates/suits/")
        
    return rank_templates, suit_templates

def recognize_card_template_matching(card_image, templates):
    """
    Template matching approach for extremely consistent card images
    
    Args:
        card_image (numpy.ndarray): Input card image
        templates (dict): Dictionary of name:template_image pairs
    
    Returns:
        str: Name of the best matching template
    """
    if not templates:
        # If no templates available, fall back to simple recognition
        if "suit" in locals():
            return simple_card_recognition(card_image)[1]  # Return suit
        else:
            return simple_card_recognition(card_image)[0]  # Return rank
        
    best_match = None
    best_score = float('-inf')
    
    for name, template in templates.items():
        if template is None:
            continue
            
        # Resize template to match card image size if needed
        template_resized = cv2.resize(template, (card_image.shape[1], card_image.shape[0]))
        
        # Match template (multiple methods available)
        result = cv2.matchTemplate(card_image, template_resized, cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = cv2.minMaxLoc(result)
        
        if score > best_score:
            best_score = score
            best_match = name
    
    # If confidence is too low, fall back to simple recognition
    if best_score < 0.5:
        print(f"Low confidence match ({best_score:.2f}). Using fallback.")
        if best_match in SUITS:
            return simple_card_recognition(card_image)[1]  # Return suit
        else:
            return simple_card_recognition(card_image)[0]  # Return rank
        
    return best_match

def predict_empty_position(preprocessed_image):
    """Predict if a position is empty using the trained model"""
    if not empty_model_loaded:
        return False  # Default to not empty if model isn't loaded
    
    # Add batch dimension
    img = np.expand_dims(preprocessed_image, axis=0)
    
    # Get prediction from empty model
    prediction = empty_model.predict(img, verbose=0)
    
    # Get the highest probability class
    is_empty = np.argmax(prediction[0]) == 0
    
    return is_empty