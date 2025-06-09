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
    empty_model_path = os.path.join(models_dir, "empty_model.h5")
    
    # Try to load rank model
    try:
        rank_model = tf.keras.models.load_model(rank_model_path)
        print(f"Successfully loaded rank model from {rank_model_path}")
    except (ImportError, FileNotFoundError):
        print(f"Error: Rank model not found at {rank_model_path}")
        return None, None, None
    
    # Try to load suit model
    try:
        suit_model = tf.keras.models.load_model(suit_model_path)
        print(f"Successfully loaded suit model from {suit_model_path}")
    except (ImportError, FileNotFoundError):
        print(f"Error: Suit model not found at {suit_model_path}")
        return rank_model, None, None
    
    # Try to load empty position model
    try:
        empty_model = tf.keras.models.load_model(empty_model_path)
        print(f"Successfully loaded empty position model from {empty_model_path}")
    except (ImportError, FileNotFoundError):
        print(f"Error: Empty position model not found at {empty_model_path}")
        return rank_model, suit_model, None
    
    return rank_model, suit_model, empty_model

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
    
    # Define rank labels to match card_recognizer.py and include '10'
    RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    
    # Define mapping from filename format to rank index
    FILENAME_TO_RANK_IDX = {
        'b_2': 0, 'r_2': 0,  # 2
        'b_3': 1, 'r_3': 1,  # 3
        'b_4': 2, 'r_4': 2,  # 4
        'b_5': 3, 'r_5': 3,  # 5
        'b_6': 4, 'r_6': 4,  # 6
        'b_7': 5, 'r_7': 5,  # 7
        'b_8': 6, 'r_8': 6,  # 8
        'b_9': 7, 'r_9': 7,  # 9
        'b_10': 8, 'r_10': 8,  # 10 -> maps to index 8
        'b_J': 9, 'r_J': 9,  # J
        'b_Q': 10, 'r_Q': 10,  # Q
        'b_K': 11, 'r_K': 11,  # K
        'b_A': 12, 'r_A': 12,  # A
    }
    
    # Get test images
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    rank_dir = os.path.join(data_dir, "cards_numbers")
    
    if not os.path.exists(rank_dir):
        print(f"Error: Directory not found: {rank_dir}")
        return
    
    # Process all rank images
    true_labels = []
    pred_labels = []
    filenames = []
    
    # Test all png images in the rank directory
    image_files = glob.glob(os.path.join(rank_dir, "*.png"))
    for image_path in image_files:
        basename = os.path.basename(image_path)
        rank_name = os.path.splitext(basename)[0]
        
        if rank_name not in FILENAME_TO_RANK_IDX:
            print(f"Warning: Unknown rank format in filename: {basename}")
            continue
            
        true_idx = FILENAME_TO_RANK_IDX[rank_name]
        filenames.append(rank_name)
        
        img = preprocess_image(image_path)
        if img is None:
            continue
        
        # Make prediction
        img_array = np.expand_dims(img, axis=0)
        prediction = model.predict(img_array, verbose=0)
        predicted_rank_idx = np.argmax(prediction[0])
        
        true_labels.append(true_idx)
        pred_labels.append(predicted_rank_idx % len(RANKS))  # Ensure index is within range
        
        print(f"Rank {rank_name}: True rank: {RANKS[true_idx]}, Predicted as {RANKS[predicted_rank_idx % len(RANKS)]}")
    
    # Calculate metrics if we have any predictions
    if true_labels and pred_labels:
        print("\n--- Rank Recognition Results ---")
        accuracy = np.mean(np.array(true_labels) == np.array(pred_labels))
        print(f"Accuracy: {accuracy:.2%}")
        
        # Create evaluation directory if it doesn't exist
        os.makedirs('evaluation', exist_ok=True)
        
        # Generate confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(true_labels, pred_labels, labels=range(len(RANKS)))
        plt.figure(figsize=(12, 10))
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
    
    # Define suit labels to match card_recognizer.py
    SUITS = ['Clubs', 'Diamonds', 'Hearts', 'Spades']
    
    # Define mapping from filename format to suit index
    FILENAME_TO_SUIT_IDX = {
        'Clubs_1': 0, 'Clubs_2': 0,
        'Diamonds_1': 1, 'Diamonds_2': 1,
        'Hearts_1': 2, 'Hearts_2': 2,
        'Spades_1': 3, 'Spades_2': 3
    }
    
    # Get test images
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    suit_dir = os.path.join(data_dir, "cards_suits")
    
    if not os.path.exists(suit_dir):
        print(f"Error: Directory not found: {suit_dir}")
        return
    
    # Process all suit images
    true_labels = []
    pred_labels = []
    filenames = []
    
    # Test all png images in the suit directory
    image_files = glob.glob(os.path.join(suit_dir, "*.png"))
    for image_path in image_files:
        basename = os.path.basename(image_path)
        suit_name = os.path.splitext(basename)[0]
        
        if suit_name not in FILENAME_TO_SUIT_IDX:
            print(f"Warning: Unknown suit format in filename: {basename}")
            continue
            
        true_idx = FILENAME_TO_SUIT_IDX[suit_name]
        filenames.append(suit_name)
        
        img = preprocess_image(image_path)
        if img is None:
            continue
        
        # Make prediction
        img_array = np.expand_dims(img, axis=0)
        prediction = model.predict(img_array, verbose=0)
        predicted_suit_idx = np.argmax(prediction[0])
        
        true_labels.append(true_idx)
        pred_labels.append(predicted_suit_idx)
        
        print(f"Suit {suit_name}: True suit: {SUITS[true_idx]}, Predicted as {SUITS[predicted_suit_idx]}")
    
    # Calculate metrics if we have any predictions
    if true_labels and pred_labels:
        print("\n--- Suit Recognition Results ---")
        accuracy = np.mean(np.array(true_labels) == np.array(pred_labels))
        print(f"Accuracy: {accuracy:.2%}")
        
        # Create evaluation directory if it doesn't exist
        os.makedirs('evaluation', exist_ok=True)
        
        # Generate confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(true_labels, pred_labels, labels=range(len(SUITS)))
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

def test_empty_model(model):
    """Test the empty position detection model"""
    if model is None:
        print("No empty position model available for testing.")
        return
    
    # Define class labels
    CLASSES = ['Empty', 'Not Empty']
    
    # Get test images
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    empty_dir = os.path.join(data_dir, "empty_positions")
    
    if not os.path.exists(empty_dir):
        print(f"Error: Directory not found: {empty_dir}")
        return
    
    # Process all empty position images
    true_labels = []
    pred_labels = []
    
    # Test empty images
    empty_images = glob.glob(os.path.join(empty_dir, "*.png"))
    for image_path in empty_images:
        img = preprocess_image(image_path)
        if img is None:
            continue
        
        # Make prediction
        img_array = np.expand_dims(img, axis=0)
        prediction = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(prediction[0])
        
        # All images in the empty_positions directory should be empty (0)
        true_labels.append(0)
        pred_labels.append(predicted_class_idx)
        
        print(f"Image {os.path.basename(image_path)}: Predicted as {CLASSES[predicted_class_idx]}")
    
    # Calculate metrics if we have any predictions
    if true_labels and pred_labels:
        print("\n--- Empty Position Detection Results ---")
        accuracy = np.mean(np.array(true_labels) == np.array(pred_labels))
        print(f"Accuracy: {accuracy:.2%}")
        
        # Create evaluation directory if it doesn't exist
        os.makedirs('evaluation', exist_ok=True)
        
        # Generate confusion matrix with explicit labels
        print("\nConfusion Matrix:")
        cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1])  # Explicitly specify all possible labels
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASSES, yticklabels=CLASSES)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Empty Position Detection Confusion Matrix')
        plt.savefig('evaluation/empty_confusion_matrix.png')
        plt.close()
        
        print(f"Confusion matrix saved to 'evaluation/empty_confusion_matrix.png'")
        
        # Classification report with explicit labels
        print("\nClassification Report:")
        try:
            # Try with explicit labels
            print(classification_report(true_labels, pred_labels, target_names=CLASSES, labels=[0, 1], zero_division=0))
        except ValueError as e:
            print(f"Could not generate complete classification report: {e}")
            # Fall back to basic accuracy
            print(f"Basic accuracy: {accuracy:.2%}")
            
            # Find which classes are present
            unique_true = np.unique(true_labels)
            unique_pred = np.unique(pred_labels)
            print(f"Classes in true labels: {[CLASSES[i] for i in unique_true]}")
            print(f"Classes in predictions: {[CLASSES[i] for i in unique_pred]}")

def main():
    """Main test function"""
    print("Testing card recognition models...")
    
    # Load models
    rank_model, suit_model, empty_model = load_models()
    
    # Test models
    print("\nTesting Rank Recognition Model:")
    test_rank_model(rank_model)
    
    print("\nTesting Suit Recognition Model:")
    test_suit_model(suit_model)
    
    print("\nTesting Empty Position Model:")
    test_empty_model(empty_model)
    
    print("\nTesting complete.")

if __name__ == "__main__":
    main()