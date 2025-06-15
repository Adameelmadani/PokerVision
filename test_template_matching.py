
import os
import cv2
import glob
import numpy as np
import argparse
import matplotlib.pyplot as plt
from card_recognizer import recognize_card_template_matching, load_card_templates

def test_template_matching():
    """
    Test the template matching method for card recognition
    """
    print("Testing template matching for card recognition...")
    
    # Load templates
    rank_templates, suit_templates = load_card_templates()
    if not rank_templates or not suit_templates:
        print("No templates found. Please run template_generator.py first.")
        return False
    
    # Set up paths
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    numbers_dir = os.path.join(data_dir, "cards_numbers")
    suits_dir = os.path.join(data_dir, "cards_suits")
    output_dir = os.path.join(os.path.dirname(__file__), "evaluation", "template_matching")
    os.makedirs(output_dir, exist_ok=True)
    
    # Define mappings
    rank_mapping = {
        'r_2': '2', 'b_2': '2',
        'r_3': '3', 'b_3': '3',
        'r_4': '4', 'b_4': '4',
        'r_5': '5', 'b_5': '5',
        'r_6': '6', 'b_6': '6',
        'r_7': '7', 'b_7': '7',
        'r_8': '8', 'b_8': '8',
        'r_9': '9', 'b_9': '9',
        'r_10': '10', 'b_10': '10',
        'r_J': 'J', 'b_J': 'J',
        'r_Q': 'Q', 'b_Q': 'Q',
        'r_K': 'K', 'b_K': 'K',
        'r_A': 'A', 'b_A': 'A'
    }
    suit_mapping = {
        'Clubs_1': 'Clubs', 'Clubs_2': 'Clubs',
        'Diamonds_1': 'Diamonds', 'Diamonds_2': 'Diamonds',
        'Hearts_1': 'Hearts', 'Hearts_2': 'Hearts',
        'Spades_1': 'Spades', 'Spades_2': 'Spades'
    }
    
    # Test rank template matching
    print("\n=== Testing Rank Template Matching ===")
    rank_correct = 0
    rank_total = 0
    rank_results = {}
    
    rank_image_files = glob.glob(os.path.join(numbers_dir, "*.png"))
    for image_path in rank_image_files:
        basename = os.path.basename(image_path)
        filename = os.path.splitext(basename)[0]
        
        if filename not in rank_mapping:
            print(f"Warning: Unknown rank format: {basename}")
            continue
            
        expected_rank = rank_mapping[filename]
        
        # Read and process image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image: {image_path}")
            continue
            
        # Run template matching
        predicted_rank = recognize_card_template_matching(img, rank_templates)
        
        # Track results
        is_correct = predicted_rank == expected_rank
        if is_correct:
            rank_correct += 1
        rank_total += 1
        
        if expected_rank not in rank_results:
            rank_results[expected_rank] = {"correct": 0, "total": 0}
        rank_results[expected_rank]["total"] += 1
        if is_correct:
            rank_results[expected_rank]["correct"] += 1
            
        print(f"Rank {filename}: Expected {expected_rank}, Predicted {predicted_rank} - {'✓' if is_correct else '✗'}")
    
    # Test suit template matching
    print("\n=== Testing Suit Template Matching ===")
    suit_correct = 0
    suit_total = 0
    suit_results = {}
    
    suit_image_files = glob.glob(os.path.join(suits_dir, "*.png"))
    for image_path in suit_image_files:
        basename = os.path.basename(image_path)
        filename = os.path.splitext(basename)[0]
        
        if filename not in suit_mapping:
            print(f"Warning: Unknown suit format: {basename}")
            continue
            
        expected_suit = suit_mapping[filename]
        
        # Read and process image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image: {image_path}")
            continue
            
        # Run template matching
        predicted_suit = recognize_card_template_matching(img, suit_templates)
        
        # Track results
        is_correct = predicted_suit == expected_suit
        if is_correct:
            suit_correct += 1
        suit_total += 1
        
        if expected_suit not in suit_results:
            suit_results[expected_suit] = {"correct": 0, "total": 0}
        suit_results[expected_suit]["total"] += 1
        if is_correct:
            suit_results[expected_suit]["correct"] += 1
            
        print(f"Suit {filename}: Expected {expected_suit}, Predicted {predicted_suit} - {'✓' if is_correct else '✗'}")
    
    # Calculate and display overall results
    print("\n=== Summary ===")
    if rank_total > 0:
        rank_accuracy = (rank_correct / rank_total) * 100
        print(f"Rank Recognition Accuracy: {rank_accuracy:.2f}% ({rank_correct}/{rank_total})")
        
        # Per-rank accuracy
        print("\nPer-Rank Accuracy:")
        for rank, result in sorted(rank_results.items()):
            if result["total"] > 0:
                accuracy = (result["correct"] / result["total"]) * 100
                print(f"  {rank}: {accuracy:.2f}% ({result['correct']}/{result['total']})")
    
    if suit_total > 0:
        suit_accuracy = (suit_correct / suit_total) * 100
        print(f"\nSuit Recognition Accuracy: {suit_accuracy:.2f}% ({suit_correct}/{suit_total})")
        
        # Per-suit accuracy
        print("\nPer-Suit Accuracy:")
        for suit, result in sorted(suit_results.items()):
            if result["total"] > 0:
                accuracy = (result["correct"] / result["total"]) * 100
                print(f"  {suit}: {accuracy:.2f}% ({result['correct']}/{result['total']})")
    
    # Create visualization if we have results
    if rank_total > 0 and suit_total > 0:
        plt.figure(figsize=(12, 6))
        
        # Rank accuracy plot
        plt.subplot(1, 2, 1)
        ranks = []
        accuracies = []
        for rank, result in sorted(rank_results.items()):
            if result["total"] > 0:
                ranks.append(rank)
                accuracies.append((result["correct"] / result["total"]) * 100)
        
        plt.bar(ranks, accuracies, color='blue')
        plt.title('Rank Recognition Accuracy')
        plt.xlabel('Rank')
        plt.ylabel('Accuracy (%)')
        plt.ylim(0, 105)
        
        # Suit accuracy plot
        plt.subplot(1, 2, 2)
        suits = []
        accuracies = []
        for suit, result in sorted(suit_results.items()):
            if result["total"] > 0:
                suits.append(suit)
                accuracies.append((result["correct"] / result["total"]) * 100)
        
        plt.bar(suits, accuracies, color='red')
        plt.title('Suit Recognition Accuracy')
        plt.xlabel('Suit')
        plt.ylabel('Accuracy (%)')
        plt.ylim(0, 105)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'template_matching_accuracy.png'))
        print(f"\nVisualization saved to {os.path.join(output_dir, 'template_matching_accuracy.png')}")
        
    return rank_total > 0 and suit_total > 0

def main():
    parser = argparse.ArgumentParser(description='Test card template matching')
    parser.add_argument('--generate', action='store_true', 
                        help='Generate templates before testing')
    args = parser.parse_args()
    
    if args.generate:
        print("Generating templates first...")
        from template_generator import generate_templates
        generate_templates()
    
    if test_template_matching():
        print("\nTemplate matching test completed successfully.")
    else:
        print("\nTemplate matching test failed. Missing templates or test images.")
        print("Run 'python template_generator.py' to create templates from your card images.")

if __name__ == "__main__":
    main()