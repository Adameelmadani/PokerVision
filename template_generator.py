
import os
import shutil
import cv2
import glob
import numpy as np

def generate_templates():
    """Generate template images for card ranks and suits from existing data"""
    print("Generating template images for card recognition...")
    
    # Define paths
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    numbers_dir = os.path.join(data_dir, "cards_numbers")
    suits_dir = os.path.join(data_dir, "cards_suits")
    templates_dir = os.path.join(data_dir, "templates")
    rank_templates_dir = os.path.join(templates_dir, "ranks")
    suit_templates_dir = os.path.join(templates_dir, "suits")
    
    # Create template directories if they don't exist
    os.makedirs(rank_templates_dir, exist_ok=True)
    os.makedirs(suit_templates_dir, exist_ok=True)
    
    # Define rank and suit mappings
    RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    SUITS = ['Clubs', 'Diamonds', 'Hearts', 'Spades']
    
    # Generate rank templates
    print("Generating rank templates...")
    for rank in RANKS:
        # Look for both red and black versions
        for color in ['r', 'b']:
            source_path = os.path.join(numbers_dir, f"{color}_{rank}.png")
            target_path = os.path.join(rank_templates_dir, f"{rank}.png")
            
            if os.path.exists(source_path) and not os.path.exists(target_path):
                print(f"Creating template for rank {rank} from {source_path}")
                # Read image and preprocess
                img = cv2.imread(source_path)
                if img is not None:
                    # Preprocessing: resize and enhance contrast for better matching
                    img = cv2.resize(img, (64, 64))
                    # Save as template
                    cv2.imwrite(target_path, img)
                    break
    
    # Generate suit templates
    print("\nGenerating suit templates...")
    for suit in SUITS:
        # Look for multiple versions of each suit
        for version in range(1, 3):  # Assuming Clubs_1.png, Clubs_2.png format
            source_path = os.path.join(suits_dir, f"{suit}_{version}.png")
            target_path = os.path.join(suit_templates_dir, f"{suit}.png")
            
            if os.path.exists(source_path) and not os.path.exists(target_path):
                print(f"Creating template for suit {suit} from {source_path}")
                # Read image and preprocess
                img = cv2.imread(source_path)
                if img is not None:
                    # Preprocessing: resize and enhance contrast for better matching
                    img = cv2.resize(img, (64, 64))
                    # Save as template
                    cv2.imwrite(target_path, img)
                    break
    
    # Count created templates
    rank_templates = glob.glob(os.path.join(rank_templates_dir, "*.png"))
    suit_templates = glob.glob(os.path.join(suit_templates_dir, "*.png"))
    
    print(f"\nTemplate generation complete.")
    print(f"Created {len(rank_templates)}/{len(RANKS)} rank templates")
    print(f"Created {len(suit_templates)}/{len(SUITS)} suit templates")
    
    if len(rank_templates) < len(RANKS) or len(suit_templates) < len(SUITS):
        print("\nMissing templates:")
        for rank in RANKS:
            if not os.path.exists(os.path.join(rank_templates_dir, f"{rank}.png")):
                print(f"  - Missing rank template: {rank}")
        for suit in SUITS:
            if not os.path.exists(os.path.join(suit_templates_dir, f"{suit}.png")):
                print(f"  - Missing suit template: {suit}")
    
    return len(rank_templates), len(suit_templates)

if __name__ == "__main__":
    num_ranks, num_suits = generate_templates()
    if num_ranks > 0 and num_suits > 0:
        print("\nYou can now use template matching: python main.py template")