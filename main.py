import time
import threading
import pygame
import os
from screenshot import take_screenshot
from card_detector import detect_cards
from card_recognizer import recognize_cards

class PokerCV:
    def __init__(self):
        self.running = False
        self.detected_cards = []
        
        # Initialize pygame for the interface
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("PokerCV - Poker Card Recognition")
        self.font = pygame.font.SysFont("Arial", 24)
        self.clock = pygame.time.Clock()
        
    def start_detection(self):
        """Start the card detection thread"""
        self.running = True
        self.detection_thread = threading.Thread(target=self._detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
    def _detection_loop(self):
        """Main loop for card detection"""
        while self.running:
            # Take a screenshot
            screenshot = take_screenshot()
            
            # Detect card regions in the screenshot
            card_regions = detect_cards(screenshot)
            
            # Recognize each card
            self.detected_cards = recognize_cards(screenshot, card_regions)
            
            # Sleep to avoid high CPU usage
            time.sleep(0.5)
            
    def run_interface(self):
        """Run the main interface loop"""
        self.start_detection()
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Clear the screen
            self.screen.fill((0, 100, 0))  # Poker table green
            
            # Display the detected cards
            y_position = 50
            self.screen.blit(self.font.render("Detected Cards:", True, (255, 255, 255)), (20, y_position))
            y_position += 40
            
            if not self.detected_cards:
                self.screen.blit(self.font.render("No cards detected", True, (255, 255, 255)), (20, y_position))
            else:
                for i, card in enumerate(self.detected_cards):
                    card_text = f"Card {i+1}: {card['rank']} of {card['suit']}"
                    self.screen.blit(self.font.render(card_text, True, (255, 255, 255)), (20, y_position))
                    y_position += 30
            
            # Update the display
            pygame.display.flip()
            self.clock.tick(30)
        
        self.running = False
        pygame.quit()

if __name__ == "__main__":
    # Create data directories if they don't exist
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    suits_dir = os.path.join(data_dir, "cards_suits")
    numbers_dir = os.path.join(data_dir, "cards_numbers")
    
    os.makedirs(suits_dir, exist_ok=True)
    os.makedirs(numbers_dir, exist_ok=True)
    
    app = PokerCV()
    app.run_interface()