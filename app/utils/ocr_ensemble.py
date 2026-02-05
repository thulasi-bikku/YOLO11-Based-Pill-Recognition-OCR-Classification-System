"""
OCR Ensemble System - Combines multiple OCR engines with weighted voting
Integrates OpenOCR, TrOCR, and PaddleOCR for robust text recognition

Created: February 5, 2026
"""

import cv2
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Optional
import yaml

from app.utils.ocr_utils import recognize_with_openocr
from app.utils.ocr_trocr import get_trocr_engine, TROCR_AVAILABLE
from app.utils.ocr_paddle import get_paddle_engine, PADDLE_AVAILABLE


class OCREnsemble:
    """
    Ensemble OCR system using multiple engines with weighted voting
    
    Combines:
    - OpenOCR (ONNX-based, fast)
    - TrOCR (transformer-based, accurate on curved text)
    - PaddleOCR (multilingual, robust)
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize OCR ensemble
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            print(f"[OCR Ensemble] Failed to load config: {e}")
            self.config = {}
        
        # Get ensemble config
        ensemble_config = self.config.get('ocr_ensemble', {})
        engines_config = ensemble_config.get('engines', {})
        
        # Engine weights
        weights = ensemble_config.get('weights', {})
        self.weights = {
            'openocr': weights.get('openocr', 1.0),
            'trocr': weights.get('trocr', 1.2),
            'paddle': weights.get('paddle', 1.0)
        }
        
        # Engine availability
        self.engines_enabled = {
            'openocr': engines_config.get('openocr', {}).get('enabled', True),
            'trocr': engines_config.get('trocr', {}).get('enabled', True) and TROCR_AVAILABLE,
            'paddle': engines_config.get('paddle', {}).get('enabled', True) and PADDLE_AVAILABLE
        }
        
        # Initialize available engines
        self._init_engines()
        
        print(f"[OCR Ensemble] Initialized with engines: {[k for k, v in self.engines_enabled.items() if v]}")
    
    def _init_engines(self):
        """Initialize OCR engines"""
        # TrOCR
        if self.engines_enabled['trocr']:
            try:
                self.trocr = get_trocr_engine()
                if self.trocr is None:
                    self.engines_enabled['trocr'] = False
            except:
                self.engines_enabled['trocr'] = False
        
        # PaddleOCR
        if self.engines_enabled['paddle']:
            try:
                self.paddle = get_paddle_engine()
                if self.paddle is None:
                    self.engines_enabled['paddle'] = False
            except:
                self.engines_enabled['paddle'] = False
    
    def recognize_ensemble(
        self,
        image: np.ndarray,
        ocr_engine=None,
        return_details: bool = False
    ) -> Tuple[List[str], float, Optional[Dict]]:
        """
        Run ensemble OCR recognition
        
        Args:
            image: OpenCV image (BGR)
            ocr_engine: OpenOCR engine instance (required for OpenOCR)
            return_details: Whether to return detailed results from each engine
        
        Returns:
            best_texts: List of top text results (ranked by confidence)
            overall_confidence: Ensemble confidence score
            details: Detailed results from each engine (if return_details=True)
        """
        results = {}
        
        # 1. OpenOCR
        if self.engines_enabled['openocr'] and ocr_engine is not None:
            try:
                texts, conf = recognize_with_openocr(
                    image,
                    ocr_engine=ocr_engine,
                    min_score=0.8
                )
                results['openocr'] = {
                    'texts': texts,
                    'confidence': conf,
                    'success': True
                }
                print(f"[OCR Ensemble] OpenOCR: {texts} (conf={conf:.3f})")
            except Exception as e:
                print(f"[OCR Ensemble] OpenOCR failed: {e}")
                results['openocr'] = {'texts': [], 'confidence': 0.0, 'success': False}
        
        # 2. TrOCR
        if self.engines_enabled['trocr']:
            try:
                text, conf = self.trocr.recognize(image, clean_text=True)
                texts = [text] if text else []
                results['trocr'] = {
                    'texts': texts,
                    'confidence': conf,
                    'success': True
                }
                print(f"[OCR Ensemble] TrOCR: {texts} (conf={conf:.3f})")
            except Exception as e:
                print(f"[OCR Ensemble] TrOCR failed: {e}")
                results['trocr'] = {'texts': [], 'confidence': 0.0, 'success': False}
        
        # 3. PaddleOCR
        if self.engines_enabled['paddle']:
            try:
                texts, conf = self.paddle.recognize(image, clean_text=True)
                results['paddle'] = {
                    'texts': texts,
                    'confidence': conf,
                    'success': True
                }
                print(f"[OCR Ensemble] PaddleOCR: {texts} (conf={conf:.3f})")
            except Exception as e:
                print(f"[OCR Ensemble] PaddleOCR failed: {e}")
                results['paddle'] = {'texts': [], 'confidence': 0.0, 'success': False}
        
        # Perform voting
        best_texts, overall_conf = self._vote(results)
        
        if return_details:
            return best_texts, overall_conf, results
        else:
            return best_texts, overall_conf, None
    
    def _vote(self, results: Dict) -> Tuple[List[str], float]:
        """
        Weighted voting across OCR engines
        
        Args:
            results: Dict with results from each engine
        
        Returns:
            best_texts: Ranked list of text candidates
            confidence: Overall confidence score
        """
        if not results:
            return [], 0.0
        
        # Collect all text candidates with weighted scores
        text_scores = Counter()
        total_weight = 0.0
        successful_engines = 0
        
        for engine_name, data in results.items():
            if not data.get('success', False):
                continue
            
            weight = self.weights.get(engine_name, 1.0)
            confidence = data.get('confidence', 0.0)
            texts = data.get('texts', [])
            
            successful_engines += 1
            total_weight += weight
            
            # Score each text
            for text in texts:
                if text:  # Non-empty text
                    score = weight * confidence
                    text_scores[text] += score
        
        # Handle no successful engines
        if successful_engines == 0 or not text_scores:
            return [], 0.0
        
        # Sort by score (highest first)
        sorted_texts = sorted(text_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Extract top texts
        best_texts = [t[0] for t in sorted_texts[:5]]  # Top 5 candidates
        
        # Normalize confidence
        max_score = sorted_texts[0][1]
        confidence = min(max_score / total_weight, 1.0) if total_weight > 0 else 0.0
        
        # Boost confidence if multiple engines agree
        if len(text_scores) > 0:
            # Check for agreement (same text from multiple engines)
            top_text = sorted_texts[0][0]
            engines_with_text = sum(
                1 for data in results.values()
                if data.get('success') and top_text in data.get('texts', [])
            )
            
            # Agreement bonus: +0.1 for each additional engine agreeing
            agreement_bonus = (engines_with_text - 1) * 0.1
            confidence = min(confidence + agreement_bonus, 1.0)
        
        return best_texts, confidence
    
    def recognize_with_rotation(
        self,
        image: np.ndarray,
        angles: List[int] = [0, 90, 180, 270],
        ocr_engine=None
    ) -> Tuple[List[str], float, int]:
        """
        Try OCR at multiple rotations, return best result
        
        Args:
            image: OpenCV image
            angles: Rotation angles to try
            ocr_engine: OpenOCR engine instance
        
        Returns:
            best_texts: Best text results
            best_confidence: Highest confidence
            best_angle: Angle that gave best result
        """
        best_texts = []
        best_confidence = 0.0
        best_angle = 0
        
        for angle in angles:
            # Rotate image
            rotated = self._rotate_image(image, angle)
            
            # Recognize
            texts, conf, _ = self.recognize_ensemble(rotated, ocr_engine)
            
            # Update best if better
            if conf > best_confidence:
                best_texts = texts
                best_confidence = conf
                best_angle = angle
        
        print(f"[OCR Ensemble] Best angle: {best_angle}° (conf={best_confidence:.3f})")
        return best_texts, best_confidence, best_angle
    
    def _rotate_image(self, image: np.ndarray, angle: int) -> np.ndarray:
        """Rotate image by given angle"""
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
    
    def get_available_engines(self) -> List[str]:
        """Get list of available/enabled engines"""
        return [k for k, v in self.engines_enabled.items() if v]
    
    def set_engine_enabled(self, engine_name: str, enabled: bool):
        """Enable/disable a specific engine"""
        if engine_name in self.engines_enabled:
            self.engines_enabled[engine_name] = enabled
            print(f"[OCR Ensemble] {engine_name} {'enabled' if enabled else 'disabled'}")


# Singleton instance
_ocr_ensemble = None


def get_ocr_ensemble(config_path: str = "config.yaml") -> OCREnsemble:
    """
    Get or create OCR ensemble singleton
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        OCREnsemble instance
    """
    global _ocr_ensemble
    if _ocr_ensemble is None:
        _ocr_ensemble = OCREnsemble(config_path=config_path)
    
    return _ocr_ensemble


# Test/Example usage
if __name__ == "__main__":
    print("[OCR Ensemble] Testing OCR Ensemble module...")
    
    try:
        ensemble = get_ocr_ensemble()
        available = ensemble.get_available_engines()
        print(f"[OCR Ensemble] ✓ Initialized with engines: {available}")
        
        if not available:
            print("[OCR Ensemble] ⚠ No OCR engines available!")
            print("[OCR Ensemble] Install dependencies:")
            print("  - TrOCR: pip install transformers torch")
            print("  - PaddleOCR: pip install paddlepaddle paddleocr")
        
    except Exception as e:
        print(f"[OCR Ensemble] ✗ Error: {e}")
