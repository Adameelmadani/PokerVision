
import cv2
import numpy as np

def detect_cards(image):
    """
    Detect card regions in the image
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        list: List of dictionaries containing card region information
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply blur and adaptive threshold
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area
    min_area = 1000  # Minimum card area, adjust based on your screen resolution
    card_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    # Extract card regions
    card_regions = []
    for i, contour in enumerate(card_contours):
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Check aspect ratio to filter out non-card contours
        aspect_ratio = float(w) / h
        if 0.5 < aspect_ratio < 0.9:  # Typical playing card aspect ratio
            card_regions.append({
                'id': i,
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'region': (x, y, w, h),
                'contour': contour
            })
    
    return card_regions

def extract_card_image(image, region):
    """
    Extract card image from the region
    
    Args:
        image (numpy.ndarray): Input image
        region (dict): Card region information
        
    Returns:
        numpy.ndarray: Extracted card image
    """
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