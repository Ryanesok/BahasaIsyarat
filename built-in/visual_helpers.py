"""
Visual UI Helpers - ROI Guide and State Indicators
"""

import cv2
import numpy as np

def draw_roi_guide(frame, color=(0, 255, 0), thickness=2):
    """
    Draw Region of Interest guide box for optimal hand placement
    """
    h, w = frame.shape[:2]
    
    # ROI dimensions (centered, 60% of frame)
    roi_w = int(w * 0.6)
    roi_h = int(h * 0.6)
    roi_x = (w - roi_w) // 2
    roi_y = (h - roi_h) // 2
    
    # Draw semi-transparent rectangle
    overlay = frame.copy()
    cv2.rectangle(overlay, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), color, thickness)
    
    # Corner markers
    corner_len = 30
    corners = [
        (roi_x, roi_y), (roi_x + roi_w, roi_y),
        (roi_x, roi_y + roi_h), (roi_x + roi_w, roi_y + roi_h)
    ]
    for cx, cy in corners:
        # Horizontal line
        cv2.line(overlay, (cx - corner_len if cx > w//2 else cx, cy),
                (cx + corner_len if cx < w//2 else cx, cy), color, thickness + 1)
        # Vertical line
        cv2.line(overlay, (cx, cy - corner_len if cy > h//2 else cy),
                (cx, cy + corner_len if cy < h//2 else cy), color, thickness + 1)
    
    # Text guide
    cv2.putText(overlay, "Place hand inside frame", (roi_x + 10, roi_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Blend
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    return frame

def draw_state_indicator(frame, state, position="top-left"):
    """
    Draw current detection state on frame
    state: "IDLE", "LISTENING", "COOLDOWN"
    """
    h, w = frame.shape[:2]
    
    state_colors = {
        "IDLE": (100, 100, 100),  # Gray
        "LISTENING": (0, 255, 0),  # Green
        "COOLDOWN": (0, 165, 255)  # Orange
    }
    
    state_text = {
        "IDLE": "[IDLE]",
        "LISTENING": "[LISTENING]",
        "COOLDOWN": "[COOLDOWN]"
    }
    
    color = state_colors.get(state, (255, 255, 255))
    text = state_text.get(state, state)
    
    # Position
    if position == "top-left":
        x, y = 10, 30
    elif position == "top-right":
        x, y = w - 150, 30
    else:
        x, y = 10, h - 30
    
    # Background box
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.rectangle(frame, (x - 5, y - text_size[1] - 5), 
                  (x + text_size[0] + 5, y + 5), (0, 0, 0), -1)
    
    # Text
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return frame
