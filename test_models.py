
import os
import numpy as np
import tensorflow as tf
import cv2
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def load_models():
    """Load the trained card recognition models"""
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    rank_model_path = os.path.join(models_dir, "rank_model.h5")
    suit_model_path = os.path.join(models_dir, "suit_model.h5")
    
    # Try to load rank model
    try:
        rank_model = tf.keras.models.load_model(rank_model_path)
        print(f"Successfully loaded rank model from {rank_model_path}")
    except (ImportError, FileNotFoundError):
        print(f"Error: Rank model not found at {rank_model_path}")
        return None, None
    
    # Try to load suit model
    try:
        suit_model = tf.keras.models.load_model(suit_model_path)
        print(f"Successfully loaded suit model from {suit_model_path}")
    except (ImportError, FileNotFoundError):
        print(f"Error: Suit model not found at {suit_model_path}")
        return rank_model, None
    
    return rank_model, suit_model

def preprocess_image(image_path):
    """Preprocess an image for model input"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error reading image: {image_path}")
            return None
        
        # Resize to the model's expected input shape
        img = cv2.resize(img, (64, 64))
        # Normalize pixel values
        img = img / 255.0
        return img
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

def test_rank_model(model):
    """Test the rank recognition model on all images in cards_numbers directory"""
    if model is None:
        print("No rank model available for testing.")
        return
    
    # Define rank labels and mapping
    RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'J', 'Q', 'K', 'A']
    RANKS_MAP = {rank: i for i, rank in enumerate(RANKS)}
    
    # Get test images
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    rank_dir = os.path.join(data_dir, "cards_numbers")
    
    if not os.path.exists(rank_dir):
        print(f"Error: Directory not found: {rank_dir}")
        return
    
    # Process all rank images
    true_labels = []
    pred_labels = []
    
    for rank in RANKS:
        image_path = os.path.join(rank_dir, f"{rank}.png")
        if not os.path.exists(image_path):
            print(f"Warning: No image found for rank {rank}")
            continue
        
        img = preprocess_image(image_path)
        if img is None:
            continue
        
        # Make prediction
        img_array = np.expand_dims(img, axis=0)
        prediction = model.predict(img_array, verbose=0)
        predicted_rank_idx = np.argmax(prediction[0])
        
        true_labels.append(RANKS_MAP[rank])
        pred_labels.append(predicted_rank_idx)
        
        print(f"Rank {rank}: Predicted as {RANKS[predicted_rank_idx]}")
    
    # Calculate metrics if we have any predictions
    if true_labels and pred_labels:
        print("\n--- Rank Recognition Results ---")
        accuracy = np.mean(np.array(true_labels) == np.array(pred_labels))
        print(f"Accuracy: {accuracy:.2%}")
        
        # Generate confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(true_labels, pred_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=RANKS, yticklabels=RANKS)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Rank Recognition Confusion Matrix')
        plt.savefig('evaluation/rank_confusion_matrix.png')
        plt.close()
        
        print(f"Confusion matrix saved to 'evaluation/rank_confusion_matrix.png'")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(true_labels, pred_labels, target_names=RANKS))

def test_suit_model(model):
    """Test the suit recognition model on all images in cards_suits directory"""
    if model is None:
        print("No suit model available for testing.")
        return
    
    # Define suit labels and mapping
    SUITS = ['Clubs', 'Diamonds', 'Hearts', 'Spades']
    SUITS_MAP = {suit: i for i, suit in enumerate(SUITS)}
    
    # Get test images
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    suit_dir = os.path.join(data_dir, "cards_suits")
    
    if not os.path.exists(suit_dir):
        print(f"Error: Directory not found: {suit_dir}")
        return
    
    # Process all suit images
    true_labels = []
    pred_labels = []
    
    for suit in SUITS:
        image_path = os.path.join(suit_dir, f"{suit}.png")
        if not os.path.exists(image_path):
            print(f"Warning: No image found for suit {suit}")
            continue
        
        img = preprocess_image(image_path)
        if img is None:
            continue
        
        # Make prediction
        img_array = np.expand_dims(img, axis=0)
        prediction = model.predict(img_array, verbose=0)
        predicted_suit_idx = np.argmax(prediction[0])
        
        true_labels.append(SUITS_MAP[suit])
        pred_labels.append(predicted_suit_idx)
        
        print(f"Suit {suit}: Predicted as {SUITS[predicted_suit_idx]}")
    
    # Calculate metrics if we have any predictions
    if true_labels and pred_labels:
        print("\n--- Suit Recognition Results ---")
        accuracy = np.mean(np.array(true_labels) == np.array(pred_labels))
        print(f"Accuracy: {accuracy:.2%}")
        
        # Generate confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(true_labels, pred_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=SUITS, yticklabels=SUITS)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Suit Recognition Confusion Matrix')
        plt.savefig('evaluation/suit_confusion_matrix.png')
        plt.close()
        
        print(f"Confusion matrix saved to 'evaluation/suit_confusion_matrix.png'")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(true_labels, pred_labels, target_names=SUITS))

def main():
    """Main test function"""
    print("Testing card recognition models...")
    
    # Load models
    rank_model, suit_model = load_models()
    
    # Test models
    print("\nTesting Rank Recognition Model:")
    test_rank_model(rank_model)
    
    print("\nTesting Suit Recognition Model:")
    test_suit_model(suit_model)
    
    print("\nTesting complete.")

if __name__ == "__main__":
    main()