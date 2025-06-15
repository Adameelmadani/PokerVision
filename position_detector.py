from screenshot import take_screenshot

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
    
    def detect_position(self):
        """
        Detect the player's position at the table.
        Returns: str - Position category ('early', 'middle', 'late', 'blind')
        """
        try:
            # Try to detect dealer button and blind indicators
            dealer_img = take_screenshot(region=self.position_regions['dealer'])
            sb_img = take_screenshot(region=self.position_regions['sb'])
            bb_img = take_screenshot(region=self.position_regions['bb'])
            
            # Calculate relative position based on dealer button
            # This is a simplified version - in practice, you'd need image processing
            # to actually detect the button and indicators
            
            # For now, return a reasonable default
            return 'middle'
            
        except Exception as e:
            print(f"Position detection error: {e}")
            return 'middle'  # Default to middle position if detection fails
