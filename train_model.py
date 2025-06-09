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
    outputs = layers.Dense(13, activation='softmax')(x)  # Changed to 13 ranks (ignoring color)
    
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
    outputs = layers.Dense(4, activation='softmax')(x)  # Changed to 4 basic suits
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_empty_position_model():
    """Create a CNN model for empty position detection"""
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
    outputs = layers.Dense(2, activation='softmax')(x)  # 2 classes: empty or not empty
    
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
        tuple: (rank_images, rank_labels, suit_images, suit_labels, empty_images, empty_labels)
    """
    # Paths for the two datasets
    ranks_path = os.path.join(base_path, "cards_numbers")
    suits_path = os.path.join(base_path, "cards_suits")
    
    # Ensure directories exist
    os.makedirs(ranks_path, exist_ok=True)
    os.makedirs(suits_path, exist_ok=True)
    
    print(f"Loading rank images from {ranks_path}")
    print(f"Loading suit images from {suits_path}")
    
    # Updated RANKS map to focus on rank only (ignoring color)
    RANKS_MAP = {
        'b_2': 0, 'b_3': 1, 'b_4': 2, 'b_5': 3, 'b_6': 4, 'b_7': 5, 'b_8': 6, 
        'b_9': 7, 'b_10': 8, 'b_J': 9, 'b_Q': 10, 'b_K': 11, 'b_A': 12,
        'r_2': 0, 'r_3': 1, 'r_4': 2, 'r_5': 3, 'r_6': 4, 'r_7': 5, 'r_8': 6,
        'r_9': 7, 'r_10': 8, 'r_J': 9, 'r_Q': 10, 'r_K': 11, 'r_A': 12
    }
    
    # Updated SUITS map to focus on basic suit only (ignoring variants)
    SUITS_MAP = {
        'Clubs_1': 0, 'Clubs_2': 0,
        'Diamonds_1': 1, 'Diamonds_2': 1,
        'Hearts_1': 2, 'Hearts_2': 2,
        'Spades_1': 3, 'Spades_2': 3
    }
    
    # Load rank images
    rank_images = []
    rank_labels = []
    
    for rank in RANKS_MAP.keys():
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
    
    # Load suit images with new naming convention
    suit_images = []
    suit_labels = []
    
    for suit in SUITS_MAP.keys():
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
        rank_images = np.random.rand(26, 64, 64, 3)  # Updated to 26 ranks
        rank_labels = np.arange(26)  # Updated to 26 ranks
    
    if len(suit_images) == 0:
        print("No suit images found! Creating dummy data for demonstration.")
        suit_images = np.random.rand(8, 64, 64, 3)  # Updated to 8 suits
        suit_labels = np.arange(8)  # Updated to 8 suits
    
    # Add loading empty position images
    empty_dir = os.path.join(base_path, "empty_positions")
    os.makedirs(empty_dir, exist_ok=True)
    print(f"Loading empty position images from {empty_dir}")
    
    # Load empty position images
    empty_images = []
    empty_labels = []
    
    # Load empty positions (class 0: empty)
    for i in range(1, 6):  # For positions 1-5
        for region_type in ["rank", "suit"]:
            image_path = os.path.join(empty_dir, f"{region_type}_pos{i}.png")
            if os.path.exists(image_path):
                try:
                    img = cv2.imread(image_path)
                    if img is not None:
                        img = cv2.resize(img, (64, 64))
                        img = img / 255.0  # Normalize
                        empty_images.append(img)
                        empty_labels.append(0)  # 0 = empty
                except Exception as e:
                    print(f"Error loading {image_path}: {e}")
    
    # Load non-empty images from rank and suit (class 1: not empty)
    # Using a subset of the rank and suit images as non-empty examples
    for img in rank_images[:min(10, len(rank_images))]:
        empty_images.append(img)
        empty_labels.append(1)  # 1 = not empty
    
    for img in suit_images[:min(10, len(suit_images))]:
        empty_images.append(img)
        empty_labels.append(1)  # 1 = not empty
    
    # Check if we have any empty position images
    if len(empty_images) == 0:
        print("No empty position images found! Creating dummy data for demonstration.")
        empty_images = np.random.rand(10, 64, 64, 3)
        empty_labels = np.zeros(10)
        
    # Convert to numpy arrays
    empty_images = np.array(empty_images)
    empty_labels = np.array(empty_labels)
    
    # Convert all data to numpy arrays and ensure matching lengths
    rank_images = np.array(rank_images)
    rank_labels = np.array(rank_labels)
    suit_images = np.array(suit_images)
    suit_labels = np.array(suit_labels)
    
    # Print data shapes to debug
    print(f"Rank images shape: {rank_images.shape}, Rank labels shape: {rank_labels.shape}")
    print(f"Suit images shape: {suit_images.shape}, Suit labels shape: {suit_labels.shape}")
    print(f"Empty images shape: {empty_images.shape}, Empty labels shape: {empty_labels.shape}")
    
    # Check if we have consistent data
    if len(rank_images) != len(rank_labels) or len(rank_images) == 0:
        print("WARNING: Rank images and labels don't match or are empty. Creating dummy data.")
        n_samples = 26
        rank_images = np.random.rand(n_samples, 64, 64, 3)
        rank_labels = np.arange(n_samples) % 26
    
    if len(suit_images) != len(suit_labels) or len(suit_images) == 0:
        print("WARNING: Suit images and labels don't match or are empty. Creating dummy data.")
        n_samples = 16
        suit_images = np.random.rand(n_samples, 64, 64, 3)
        suit_labels = np.arange(n_samples) % 8
    
    if len(empty_images) != len(empty_labels) or len(empty_images) == 0:
        print("WARNING: Empty position images and labels don't match. Creating dummy data.")
        n_samples = 10
        empty_images = np.random.rand(n_samples, 64, 64, 3)
        empty_labels = np.zeros(n_samples)
        # Add some non-empty examples
        empty_labels[:5] = 1
    
    return rank_images, rank_labels, suit_images, suit_labels, empty_images, empty_labels

def train():
    """Train separate models for card rank, suit recognition and empty position detection"""
    # Ensure model directory exists
    os.makedirs(os.path.join(os.path.dirname(__file__), "models"), exist_ok=True)
    rank_model_path = os.path.join(os.path.dirname(__file__), "models", "rank_model.h5")
    suit_model_path = os.path.join(os.path.dirname(__file__), "models", "suit_model.h5")
    empty_model_path = os.path.join(os.path.dirname(__file__), "models", "empty_model.h5")
    
    # Load datasets
    dataset_path = os.path.join(os.path.dirname(__file__), "data")
    rank_images, rank_labels, suit_images, suit_labels, empty_images, empty_labels = load_datasets(dataset_path)
    
    # Split rank dataset into training and testing sets
    X_rank_train, X_rank_test, y_rank_train, y_rank_test = train_test_split(
        rank_images, rank_labels, test_size=0.2, random_state=42
    )
    
    # Split suit dataset into training and testing sets
    X_suit_train, X_suit_test, y_suit_train, y_suit_test = train_test_split(
        suit_images, suit_labels, test_size=0.2, random_state=42
    )
    
    X_empty_train, X_empty_test, y_empty_train, y_empty_test = train_test_split(
        empty_images, empty_labels, test_size=0.2, random_state=42
    )
    
    # Convert labels to one-hot encoding
    y_rank_train = tf.keras.utils.to_categorical(y_rank_train, 13)  # Changed to 13 ranks
    y_rank_test = tf.keras.utils.to_categorical(y_rank_test, 13)  # Changed to 13 ranks
    y_suit_train = tf.keras.utils.to_categorical(y_suit_train, 4)  # Changed to 4 suits
    y_suit_test = tf.keras.utils.to_categorical(y_suit_test, 4)  # Changed to 4 suits
    y_empty_train = tf.keras.utils.to_categorical(y_empty_train, 2)
    y_empty_test = tf.keras.utils.to_categorical(y_empty_test, 2)
    
    # Print dataset shapes before training
    print(f"Training data shapes:")
    print(f"X_rank_train: {X_rank_train.shape}, y_rank_train: {y_rank_train.shape}")
    print(f"X_suit_train: {X_suit_train.shape}, y_suit_train: {y_suit_train.shape}")
    print(f"X_empty_train: {X_empty_train.shape}, y_empty_train: {y_empty_train.shape}")
    
    # Create and train rank model
    print("Training rank recognition model...")
    rank_model = create_rank_model()
    rank_model.fit(
        X_rank_train, y_rank_train,
        epochs=300,
        batch_size=12,
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
    
    # Create and train empty position model
    print("Training empty position detection model...")
    empty_model = create_empty_position_model()
    empty_model.fit(
        X_empty_train, y_empty_train,
        epochs=100,
        batch_size=8,
        validation_data=(X_empty_test, y_empty_test)
    )
    
    # Save the models
    rank_model.save(rank_model_path)
    suit_model.save(suit_model_path)
    empty_model.save(empty_model_path)
    print(f"Rank model saved to {rank_model_path}")
    print(f"Suit model saved to {suit_model_path}")
    print(f"Empty position model saved to {empty_model_path}")

if __name__ == "__main__":
    train()