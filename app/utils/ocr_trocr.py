"""
TrOCR Engine - Transformer-based OCR for Pill Text Recognition
Handles curved text and complex fonts better than traditional OCR

Created: February 5, 2026
"""

import cv2
import numpy as np
import re
from typing import Tuple, Optional

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from PIL import Image
    import torch
    TROCR_AVAILABLE = True
except ImportError:
    print("[TrOCR] transformers or torch not installed. TrOCR features disabled.")
    print("[TrOCR] Install with: pip install transformers torch pillow")
    TROCR_AVAILABLE = False


class TrOCREngine:
    """
    TrOCR engine for transformer-based OCR
    Superior performance on curved and stylized text
    """
    
    def __init__(self, model_name: str = "microsoft/trocr-base-printed"):
        """
        Initialize TrOCR model
        
        Args:
            model_name: Hugging Face model name
                       Options:
                       - microsoft/trocr-base-printed (recommended for pills)
                       - microsoft/trocr-small-printed (faster, less accurate)
                       - microsoft/trocr-base-handwritten (for handwritten text)
        """
        if not TROCR_AVAILABLE:
            raise ImportError("transformers and torch are required for TrOCR")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        print(f"[TrOCR] Loading {model_name} on {self.device}...")
        
        try:
            # Load processor and model
            self.processor = TrOCRProcessor.from_pretrained(model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            print(f"[TrOCR] Model ready on {self.device}")
        except Exception as e:
            print(f"[TrOCR] Failed to load model: {e}")
            raise
    
    def recognize(self, image_cv: np.ndarray, clean_text: bool = True) -> Tuple[str, float]:
        """
        Recognize text from image using TrOCR
        
        Args:
            image_cv: OpenCV image (BGR numpy array)
            clean_text: Whether to clean and uppercase the text
        
        Returns:
            text: Recognized text string
            confidence: Confidence score (approximated)
        """
        try:
            # Convert BGR to RGB PIL Image
            image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # Process image
            pixel_values = self.processor(
                pil_image,
                return_tensors="pt"
            ).pixel_values.to(self.device)
            
            # Generate text with attention scores
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values)
            
            # Decode text
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Clean text if requested
            if clean_text:
                text = self._clean_text(text)
            
            # Approximate confidence (TrOCR doesn't provide direct confidence)
            # Higher confidence for longer, alphanumeric text
            confidence = self._estimate_confidence(text)
            
            return text, confidence
            
        except Exception as e:
            print(f"[TrOCR] Recognition failed: {e}")
            return "", 0.0
    
    def recognize_batch(self, images: list, clean_text: bool = True) -> list:
        """
        Recognize text from multiple images (batch processing)
        
        Args:
            images: List of OpenCV images
            clean_text: Whether to clean text
        
        Returns:
            List of (text, confidence) tuples
        """
        results = []
        
        # Convert all to PIL images
        pil_images = []
        for img in images:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_images.append(Image.fromarray(img_rgb))
        
        try:
            # Process batch
            pixel_values = self.processor(
                pil_images,
                return_tensors="pt"
            ).pixel_values.to(self.device)
            
            # Generate for batch
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values)
            
            # Decode all
            texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Process results
            for text in texts:
                if clean_text:
                    text = self._clean_text(text)
                confidence = self._estimate_confidence(text)
                results.append((text, confidence))
            
        except Exception as e:
            print(f"[TrOCR] Batch recognition failed: {e}")
            results = [("", 0.0) for _ in images]
        
        return results
    
    def recognize_rotated(
        self,
        image_cv: np.ndarray,
        angles: list = [0, 90, 180, 270],
        clean_text: bool = True
    ) -> Tuple[str, float, int]:
        """
        Try OCR at multiple rotations and return best result
        
        Args:
            image_cv: OpenCV image
            angles: List of rotation angles to try
            clean_text: Whether to clean text
        
        Returns:
            best_text: Best recognized text
            best_confidence: Highest confidence
            best_angle: Angle that gave best result
        """
        best_text = ""
        best_confidence = 0.0
        best_angle = 0
        
        for angle in angles:
            # Rotate image
            rotated = self._rotate_image(image_cv, angle)
            
            # Recognize
            text, conf = self.recognize(rotated, clean_text=clean_text)
            
            # Update best if better
            if conf > best_confidence:
                best_text = text
                best_confidence = conf
                best_angle = angle
        
        return best_text, best_confidence, best_angle
    
    def _clean_text(self, text: str) -> str:
        """
        Clean OCR text output for pill identification
        - Convert to uppercase
        - Keep only alphanumeric and hyphens
        - Strip whitespace
        """
        # Convert to uppercase
        text = text.upper()
        
        # Remove non-alphanumeric except hyphens
        text = re.sub(r'[^A-Z0-9\\-]', '', text)
        
        # Remove extra hyphens
        text = re.sub(r'-+', '-', text)
        
        return text.strip()
    
    def _estimate_confidence(self, text: str) -> float:
        """
        Estimate confidence based on text characteristics
        (TrOCR doesn't provide direct confidence scores)
        """
        if not text:
            return 0.0
        
        # Start with base confidence
        confidence = 0.75
        
        # Longer text = higher confidence
        if len(text) >= 3:
            confidence += 0.1
        if len(text) >= 5:
            confidence += 0.05
        
        # Mostly alphanumeric = higher confidence
        alphanumeric_ratio = sum(c.isalnum() for c in text) / len(text)
        confidence += alphanumeric_ratio * 0.1
        
        # Cap at 0.95 (never claim 100% confidence)
        return min(confidence, 0.95)
    
    def _rotate_image(self, image: np.ndarray, angle: int) -> np.ndarray:
        """Rotate image by given angle (0, 90, 180, 270)"""
        if angle == 0:
            return image
        elif angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            # Custom rotation
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            return rotated


# Singleton instance
_trocr_engine = None


def get_trocr_engine(model_name: str = "microsoft/trocr-base-printed") -> Optional[TrOCREngine]:
    """
    Get or create TrOCR engine singleton
    
    Args:
        model_name: Model to use
    
    Returns:
        TrOCREngine instance or None if unavailable
    """
    if not TROCR_AVAILABLE:
        return None
    
    global _trocr_engine
    if _trocr_engine is None:
        try:
            _trocr_engine = TrOCREngine(model_name=model_name)
        except Exception as e:
            print(f"[TrOCR] Failed to initialize: {e}")
            return None
    
    return _trocr_engine


# Test/Example usage
if __name__ == "__main__":
    print("[TrOCR] Testing TrOCR module...")
    
    if TROCR_AVAILABLE:
        print("[TrOCR] ✓ Dependencies available")
        
        # Try to create engine
        try:
            engine = get_trocr_engine()
            if engine:
                print(f"[TrOCR] ✓ Engine initialized on {engine.device}")
            else:
                print("[TrOCR] ✗ Failed to initialize engine")
        except Exception as e:
            print(f"[TrOCR] ✗ Error: {e}")
    else:
        print("[TrOCR] ✗ Dependencies NOT available")
        print("[TrOCR] Install with: pip install transformers torch pillow")
