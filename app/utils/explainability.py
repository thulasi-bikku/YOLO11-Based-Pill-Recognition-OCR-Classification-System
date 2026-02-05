"""
Explainable AI (XAI) Module for Pill Detection
Provides Grad-CAM and other visualization techniques for YOLO models

Created: February 5, 2026
"""

import cv2
import torch
import numpy as np
from typing import List, Tuple, Optional
import yaml

try:
    from pytorch_grad_cam import GradCAM, EigenCAM, GradCAMPlusPlus
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    print("[XAI] pytorch-grad-cam not installed. XAI features disabled.")
    print("[XAI] Install with: pip install pytorch-grad-cam")
    GRADCAM_AVAILABLE = False


class YOLOGradCAM:
    """
    Generate Grad-CAM heatmaps for YOLO detections
    Visualizes which regions of the image contributed to pill detection
    """
    
    def __init__(self, model, method="EigenCAM", target_layers=None):
        """
        Initialize Grad-CAM for YOLO model
        
        Args:
            model: YOLO model instance
            method: CAM method - "GradCAM", "EigenCAM", or "GradCAMPlusPlus"
            target_layers: Specific layers to target (auto-detect if None)
        """
        if not GRADCAM_AVAILABLE:
            raise ImportError("pytorch-grad-cam is required for XAI features")
        
        self.model = model
        self.method = method
        
        # Auto-detect target layers from YOLO architecture
        if target_layers is None:
            # YOLO11/YOLOv8 last backbone layer before head
            try:
                self.target_layers = [model.model.model[-4]]  # Try backbone last layer
            except:
                # Fallback to model[-2]
                self.target_layers = [model.model.model[-2]]
        else:
            self.target_layers = target_layers
        
        # Select CAM method
        if method == "GradCAM":
            self.cam_class = GradCAM
        elif method == "EigenCAM":
            self.cam_class = EigenCAM
        elif method == "GradCAMPlusPlus":
            self.cam_class = GradCAMPlusPlus
        else:
            print(f"[XAI] Unknown method {method}, using EigenCAM")
            self.cam_class = EigenCAM
    
    def generate_heatmap(
        self,
        image: np.ndarray,
        detection_box: Optional[Tuple[int, int, int, int]] = None,
        colormap: str = "jet",
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for image
        
        Args:
            image: Original image (BGR numpy array)
            detection_box: [x1, y1, x2, y2] bounding box (optional, for cropping)
            colormap: OpenCV colormap name
            alpha: Overlay transparency (0=invisible, 1=opaque)
        
        Returns:
            heatmap_overlay: Image with heatmap overlay (BGR)
        """
        # Prepare image
        if detection_box is not None:
            x1, y1, x2, y2 = detection_box
            cropped_img = image[y1:y2, x1:x2].copy()
        else:
            cropped_img = image.copy()
        
        # Resize to YOLO input size
        img_resized = cv2.resize(cropped_img, (640, 640))
        
        # Convert to RGB for processing
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB) / 255.0
        
        # Prepare tensor
        img_tensor = torch.from_numpy(img_rgb).float()
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # CHW, add batch
        
        # Create CAM
        cam = self.cam_class(self.model.model, self.target_layers)
        
        try:
            # Generate heatmap
            grayscale_cam = cam(input_tensor=img_tensor)[0, :]
            
            # Overlay on image
            heatmap_overlay = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=False)
            
            # Convert back to BGR
            heatmap_overlay = cv2.cvtColor(heatmap_overlay, cv2.COLOR_RGB2BGR)
            
            # Resize back to original crop size
            if detection_box is not None:
                heatmap_overlay = cv2.resize(heatmap_overlay, (x2 - x1, y2 - y1))
            else:
                heatmap_overlay = cv2.resize(heatmap_overlay, (image.shape[1], image.shape[0]))
            
            return heatmap_overlay
            
        except Exception as e:
            print(f"[XAI] Heatmap generation failed: {e}")
            return image  # Return original on failure
    
    def generate_multiple_heatmaps(
        self,
        image: np.ndarray,
        boxes: List[Tuple[int, int, int, int]]
    ) -> List[np.ndarray]:
        """
        Generate heatmaps for multiple detections
        
        Args:
            image: Original image
            boxes: List of bounding boxes [[x1, y1, x2, y2], ...]
        
        Returns:
            List of heatmap overlays
        """
        heatmaps = []
        for box in boxes:
            heatmap = self.generate_heatmap(image, box)
            heatmaps.append(heatmap)
        return heatmaps


def add_xai_visualization(
    image: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    model,
    method: str = "EigenCAM",
    config: dict = None
) -> List[np.ndarray]:
    """
    Convenience function to add XAI heatmaps to detection results
    
    Args:
        image: Original image (BGR)
        boxes: Detection bounding boxes
        model: YOLO model instance
        method: CAM method
        config: Configuration dict (optional)
    
    Returns:
        List of heatmap visualizations
    """
    if not GRADCAM_AVAILABLE:
        print("[XAI] Grad-CAM not available, returning empty list")
        return []
    
    # Load config if provided
    if config is None:
        try:
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
        except:
            config = {}
    
    # Check if XAI is enabled
    if not config.get("xai", {}).get("enabled", True):
        return []
    
    # Get XAI settings
    xai_config = config.get("xai", {})
    method = xai_config.get("method", method)
    
    try:
        xai = YOLOGradCAM(model, method=method)
        heatmaps = xai.generate_multiple_heatmaps(image, boxes)
        return heatmaps
    except Exception as e:
        print(f"[XAI] Visualization failed: {e}")
        return []


def visualize_detection_with_xai(
    image: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    labels: List[str],
    model,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Create comprehensive visualization with bounding boxes and heatmaps
    
    Args:
        image: Original image
        boxes: Detection boxes
        labels: Detection labels
        model: YOLO model
        save_path: Path to save visualization (optional)
    
    Returns:
        Annotated image
    """
    result_img = image.copy()
    
    # Generate heatmaps
    heatmaps = add_xai_visualization(image, boxes, model)
    
    # Draw boxes and labels
    for idx, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = map(int, box)
        
        # Draw bounding box
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label_text = f"{label}"
        cv2.putText(
            result_img,
            label_text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )
        
        # Overlay heatmap if available
        if heatmaps and idx < len(heatmaps):
            heatmap = heatmaps[idx]
            # Blend heatmap with original region
            roi = result_img[y1:y2, x1:x2]
            blended = cv2.addWeighted(roi, 0.6, heatmap, 0.4, 0)
            result_img[y1:y2, x1:x2] = blended
    
    # Save if path provided
    if save_path:
        cv2.imwrite(save_path, result_img)
        print(f"[XAI] Saved visualization to {save_path}")
    
    return result_img


# Example usage
if __name__ == "__main__":
    print("[XAI] Testing XAI module...")
    
    if GRADCAM_AVAILABLE:
        print("[XAI] ✓ pytorch-grad-cam is available")
    else:
        print("[XAI] ✗ pytorch-grad-cam is NOT available")
        print("[XAI] Install with: pip install pytorch-grad-cam")
