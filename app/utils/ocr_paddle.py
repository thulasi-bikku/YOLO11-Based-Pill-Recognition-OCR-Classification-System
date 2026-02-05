"""
PaddleOCR Engine - High-performance multilingual OCR
Complementary to OpenOCR and TrOCR for ensemble voting

Created: February 5, 2026
"""

import cv2
import numpy as np
import re
from typing import List, Tuple, Optional

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    print("[PaddleOCR] paddleocr not installed. PaddleOCR features disabled.")
    print("[PaddleOCR] Install with: pip install paddlepaddle paddleocr")
    PADDLE_AVAILABLE = False


class PaddleOCREngine:
    """
    PaddleOCR engine wrapper for pill text recognition
    Fast and accurate multilingual OCR
    """
    
    def __init__(
        self,
        use_angle_cls: bool = True,
        lang: str = 'en',
        use_gpu: bool = False,
        show_log: bool = False
    ):
        """
        Initialize PaddleOCR
        
        Args:
            use_angle_cls: Enable angle classification (auto-rotation)
            lang: Language code ('en', 'ch', etc.)
            use_gpu: Use GPU if available
            show_log: Show detailed logging
        """
        if not PADDLE_AVAILABLE:
            raise ImportError("paddleocr is required. Install with: pip install paddleocr paddlepaddle")
        
        self.lang = lang
        
        print(f"[PaddleOCR] Initializing (lang={lang}, gpu={use_gpu})...")
        
        try:
            self.ocr = PaddleOCR(
                use_angle_cls=use_angle_cls,
                lang=lang,
                use_gpu=use_gpu,
                show_log=show_log
            )
            print("[PaddleOCR] Ready")
        except Exception as e:
            print(f"[PaddleOCR] Failed to initialize: {e}")
            raise
    
    def recognize(
        self,
        image_cv: np.ndarray,
        cls: bool = True,
        clean_text: bool = True
    ) -> Tuple[List[str], float]:
        """
        Recognize text from image
        
        Args:
            image_cv: OpenCV image (BGR numpy array)
            cls: Enable angle classification
            clean_text: Clean and uppercase text
        
        Returns:
            texts: List of recognized text strings
            avg_confidence: Average confidence score
        """
        try:
            # Run OCR
            result = self.ocr.ocr(image_cv, cls=cls)
            
            # Check if result is valid
            if not result or not result[0]:
                return [], 0.0
            
            texts = []
            confidences = []
            
            # Extract text and confidence from each line
            for line in result[0]:
                # Line format: [box_coordinates, (text, confidence)]
                text = line[1][0]
                conf = line[1][1]
                
                # Clean text if requested
                if clean_text:
                    text = self._clean_text(text)
                
                # Only add non-empty text
                if text:
                    texts.append(text)
                    confidences.append(conf)
            
            # Calculate average confidence
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            
            return texts, avg_conf
            
        except Exception as e:
            print(f"[PaddleOCR] Recognition failed: {e}")
            return [], 0.0
    
    def recognize_with_boxes(
        self,
        image_cv: np.ndarray,
        cls: bool = True,
        clean_text: bool = True
    ) -> Tuple[List[dict], float]:
        """
        Recognize text with bounding box information
        
        Args:
            image_cv: OpenCV image
            cls: Enable angle classification
            clean_text: Clean text
        
        Returns:
            results: List of dicts with {text, confidence, box}
            avg_confidence: Average confidence
        """
        try:
            result = self.ocr.ocr(image_cv, cls=cls)
            
            if not result or not result[0]:
                return [], 0.0
            
            results = []
            confidences = []
            
            for line in result[0]:
                box = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                text = line[1][0]
                conf = line[1][1]
                
                if clean_text:
                    text = self._clean_text(text)
                
                if text:
                    results.append({
                        'text': text,
                        'confidence': conf,
                        'box': box
                    })
                    confidences.append(conf)
            
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            
            return results, avg_conf
            
        except Exception as e:
            print(f"[PaddleOCR] Recognition with boxes failed: {e}")
            return [], 0.0
    
    def recognize_detection_only(self, image_cv: np.ndarray) -> List[np.ndarray]:
        """
        Perform text detection only (no recognition)
        Returns cropped text regions
        
        Args:
            image_cv: OpenCV image
        
        Returns:
            List of cropped text regions
        """
        try:
            # Use detection only
            result = self.ocr.ocr(image_cv, det=True, rec=False)
            
            if not result or not result[0]:
                return []
            
            crops = []
            for box in result[0]:
                # Get bounding box coordinates
                pts = np.array(box, dtype=np.int32)
                x_min = pts[:, 0].min()
                y_min = pts[:, 1].min()
                x_max = pts[:, 0].max()
                y_max = pts[:, 1].max()
                
                # Crop region
                crop = image_cv[y_min:y_max, x_min:x_max]
                crops.append(crop)
            
            return crops
            
        except Exception as e:
            print(f"[PaddleOCR] Detection only failed: {e}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """
        Clean OCR text for pill identification
        - Convert to uppercase
        - Keep only alphanumeric and hyphens
        - Strip whitespace
        """
        # Convert to uppercase
        text = text.upper()
        
        # Remove non-alphanumeric except hyphens
        text = re.sub(r'[^A-Z0-9\\-]', '', text)
        
        # Remove multiple hyphens
        text = re.sub(r'-+', '-', text)
        
        return text.strip()


# Singleton instance
_paddle_engine = None


def get_paddle_engine(
    use_angle_cls: bool = True,
    lang: str = 'en',
    use_gpu: bool = False
) -> Optional[PaddleOCREngine]:
    """
    Get or create PaddleOCR engine singleton
    
    Args:
        use_angle_cls: Enable angle classification
        lang: Language code
        use_gpu: Use GPU
    
    Returns:
        PaddleOCREngine instance or None if unavailable
    """
    if not PADDLE_AVAILABLE:
        return None
    
    global _paddle_engine
    if _paddle_engine is None:
        try:
            _paddle_engine = PaddleOCREngine(
                use_angle_cls=use_angle_cls,
                lang=lang,
                use_gpu=use_gpu,
                show_log=False
            )
        except Exception as e:
            print(f"[PaddleOCR] Failed to initialize: {e}")
            return None
    
    return _paddle_engine


# Test/Example usage
if __name__ == "__main__":
    print("[PaddleOCR] Testing PaddleOCR module...")
    
    if PADDLE_AVAILABLE:
        print("[PaddleOCR] ✓ Dependencies available")
        
        # Try to create engine
        try:
            engine = get_paddle_engine()
            if engine:
                print("[PaddleOCR] ✓ Engine initialized")
            else:
                print("[PaddleOCR] ✗ Failed to initialize engine")
        except Exception as e:
            print(f"[PaddleOCR] ✗ Error: {e}")
    else:
        print("[PaddleOCR] ✗ Dependencies NOT available")
        print("[PaddleOCR] Install with: pip install paddlepaddle paddleocr")
