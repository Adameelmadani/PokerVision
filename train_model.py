import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Model, Input
import cv2
from sklearn.model_selection import train_test_split
import glob

def create_rank_model():
    """Create a CNN model for card rank recognition"""
    inputs = Input(shape=(64, 64, 3))
    
    # Feature extraction
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Classification
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(12, activation='softmax')(x)  # Changed from 13 to 12 ranks
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_suit_model():
    """Create a CNN model for card suit recognition"""
    inputs = Input(shape=(64, 64, 3))
    
    # Feature extraction
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Classification
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(4, activation='softmax')(x)  # 4 suits
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_datasets(base_path):
    """
    Load and preprocess the card datasets from separate folders
    
    Args:
        base_path (str): Base path to dataset directories
        
    Returns:
        tuple: (rank_images, rank_labels, suit_images, suit_labels)
    """
    # Paths for the two datasets
    ranks_path = os.path.join(base_path, "cards_numbers")
    suits_path = os.path.join(base_path, "cards_suits")
    
    # Ensure directories exist
    os.makedirs(ranks_path, exist_ok=True)
    os.makedirs(suits_path, exist_ok=True)
    
    print(f"Loading rank images from {ranks_path}")
    print(f"Loading suit images from {suits_path}")
    
    # RANKS maps to index (removed '10')
    RANKS_MAP = {
        '2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 
        'J': 8, 'Q': 9, 'K': 10, 'A': 11
    }
    
    # SUITS maps to index
    SUITS_MAP = {
        'Clubs': 0, 'Diamonds': 1, 'Hearts': 2, 'Spades': 3
    }
    
    # Load rank images
    rank_images = []
    rank_labels = []
    
    for rank in RANKS_MAP.keys():
        # Changed from f"{rank}.*" to f"{rank}.png"
        rank_image_path = os.path.join(ranks_path, f"{rank}.png")
        image_files = glob.glob(rank_image_path)
        
        if not image_files:
            print(f"Warning: No PNG image found for rank {rank}")
            continue
            
        for img_file in image_files:
            try:
                img = cv2.imread(img_file)
                if img is not None:
                    img = cv2.resize(img, (64, 64))
                    img = img / 255.0  # Normalize
                    rank_images.append(img)
                    rank_labels.append(RANKS_MAP[rank])
            except Exception as e:
                print(f"Error loading {img_file}: {e}")
    
    # Load suit images
    suit_images = []
    suit_labels = []
    
    for suit in SUITS_MAP.keys():
        # Changed from f"{suit}.*" to f"{suit}.png"
        suit_image_path = os.path.join(suits_path, f"{suit}.png")
        image_files = glob.glob(suit_image_path)
        
        if not image_files:
            print(f"Warning: No PNG image found for suit {suit}")
            continue
            
        for img_file in image_files:
            try:
                img = cv2.imread(img_file)
                if img is not None:
                    img = cv2.resize(img, (64, 64))
                    img = img / 255.0  # Normalize
                    suit_images.append(img)
                    suit_labels.append(SUITS_MAP[suit])
            except Exception as e:
                print(f"Error loading {img_file}: {e}")
    
    # Check if we have any images
    if len(rank_images) == 0:
        print("No rank images found! Creating dummy data for demonstration.")
        rank_images = np.random.rand(12, 64, 64, 3)  # Changed from 13 to 12
        rank_labels = np.arange(12)  # Changed from 13 to 12
    
    if len(suit_images) == 0:
        print("No suit images found! Creating dummy data for demonstration.")
        suit_images = np.random.rand(4, 64, 64, 3)
        suit_labels = np.arange(4)
    
    # Convert to numpy arrays
    rank_images = np.array(rank_images)
    rank_labels = np.array(rank_labels)
    suit_images = np.array(suit_images)
    suit_labels = np.array(suit_labels)
    
    return rank_images, rank_labels, suit_images, suit_labels

def train():
    """Train separate models for card rank and suit recognition"""
    # Ensure model directory exists
    os.makedirs(os.path.join(os.path.dirname(__file__), "models"), exist_ok=True)
    rank_model_path = os.path.join(os.path.dirname(__file__), "models", "rank_model.h5")
    suit_model_path = os.path.join(os.path.dirname(__file__), "models", "suit_model.h5")
    
    # Load datasets
    dataset_path = os.path.join(os.path.dirname(__file__), "data")
    rank_images, rank_labels, suit_images, suit_labels = load_datasets(dataset_path)
    
    # Split rank dataset into training and testing sets
    X_rank_train, X_rank_test, y_rank_train, y_rank_test = train_test_split(
        rank_images, rank_labels, test_size=0.2, random_state=42
    )
    
    # Split suit dataset into training and testing sets
    X_suit_train, X_suit_test, y_suit_train, y_suit_test = train_test_split(
        suit_images, suit_labels, test_size=0.2, random_state=42
    )
    
    # Convert labels to one-hot encoding
    y_rank_train = tf.keras.utils.to_categorical(y_rank_train, 12)  # Changed from 13 to 12
    y_rank_test = tf.keras.utils.to_categorical(y_rank_test, 12)  # Changed from 13 to 12
    y_suit_train = tf.keras.utils.to_categorical(y_suit_train, 4)
    y_suit_test = tf.keras.utils.to_categorical(y_suit_test, 4)
    
    # Create and train rank model
    print("Training rank recognition model...")
    rank_model = create_rank_model()
    rank_model.fit(
        X_rank_train, y_rank_train,
        epochs=300,
        batch_size=8,
        validation_data=(X_rank_test, y_rank_test)
    )
    
    # Create and train suit model
    print("Training suit recognition model...")
    suit_model = create_suit_model()
    suit_model.fit(
        X_suit_train, y_suit_train,
        epochs=300,
        batch_size=8,
        validation_data=(X_suit_test, y_suit_test)
    )
    
    # Save the models
    rank_model.save(rank_model_path)
    suit_model.save(suit_model_path)
    print(f"Rank model saved to {rank_model_path}")
    print(f"Suit model saved to {suit_model_path}")

if __name__ == "__main__":
    train()