import cv2
import numpy as np
from screenshot import take_screenshot
from card_recognizer import preprocess_card_image

class PositionDetector:
    def __init__(self):
        # Define regions for position indicators
        # These coordinates should be adjusted based on the PokerStars layout
        self.position_regions = {
            'dealer': (550, 350, 20, 20),    # Dealer button region
            'sb': (500, 350, 20, 20),        # Small blind indicator region
            'bb': (450, 350, 20, 20),        # Big blind indicator region
        }
        
        # Position mapping based on 6-max table
        self.position_map = {
            0: 'early',    # UTG
            1: 'early',    # UTG+1
            2: 'middle',   # MP
            3: 'late',     # CO
            4: 'late',     # BTN
            5: 'blind'     # BB
        }
    
    def _detect_circle(self, image):
        """
        Detect circular buttons/indicators in the image.
        Returns: bool - True if a circular indicator is detected
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply blur and threshold
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check each contour
        for contour in contours:
            # Get shape properties
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            
            # Skip small contours
            if area < 50:  # Adjust based on your button size
                continue
                
            # Calculate circularity
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # If shape is roughly circular (circularity close to 1)
            if 0.8 < circularity < 1.2:
                return True
                
        return False
    
    def detect_position(self):
        """
        Detect the player's position at the table.
        Returns: str - Position category ('early', 'middle', 'late', 'blind')
        """
        try:
            # Take screenshots of indicator regions
            dealer_img = take_screenshot(region=self.position_regions['dealer'])
            sb_img = take_screenshot(region=self.position_regions['sb'])
            bb_img = take_screenshot(region=self.position_regions['bb'])
            
            # Detect indicators
            dealer_present = self._detect_circle(dealer_img)
            sb_present = self._detect_circle(sb_img)
            bb_present = self._detect_circle(bb_img)
            
            # Calculate relative position based on dealer button and blind indicators
            if bb_present:  # Player is in big blind
                return 'blind'
            elif sb_present:  # Player is in small blind
                return 'blind'
            elif dealer_present:  # Player has the dealer button
                return 'late'  # BTN is always late position
            else:
                # Need to detect relative position based on dealer button location
                # For now, default to middle position if unsure
                return 'middle'
            
        except Exception as e:
            print(f"Position detection error: {e}")
            return 'middle'  # Default to middle position if detection fails
