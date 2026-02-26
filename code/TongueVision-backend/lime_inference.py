import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor
from PIL import Image
import numpy as np
import io
import base64
from lime import lime_image
from skimage.segmentation import mark_boundaries, slic
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = (224, 224)  # Updated to match your V6 test script
MODEL_PATH = 'DentalLens_V6.pth'
NUM_CLASSES = 6

# ⚠️ UPDATE THIS ARRAY with the exact output of `test_dataset.classes` from your test script
CLASSES = ['calculus', 'caries', 'gingivitis', 'hypodontia', 'mouth ulcer', 'tooth discoloration'] 

# ==================== MODEL ARCHITECTURE ====================
class AGFFBlock(nn.Module):
    def __init__(self, in_channels=768):
        super(AGFFBlock, self).__init__()
        self.half_channels = in_channels // 2
        
        self.ln_conv = nn.LayerNorm(in_channels)
        self.ln_swin = nn.LayerNorm(in_channels)
        self.proj_conv = nn.Conv2d(in_channels, self.half_channels, kernel_size=1)
        self.proj_swin = nn.Conv2d(in_channels, self.half_channels, kernel_size=1)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        reduction_dim = max(in_channels // 16, 32)
        self.channel_mlp = nn.Sequential(
            nn.Linear(in_channels, reduction_dim),
            nn.ReLU(),
            nn.Linear(reduction_dim, in_channels),
            nn.Sigmoid()
        )

    def forward(self, f_conv, f_swin):
        if f_conv.shape[2:] != f_swin.shape[2:]:
            f_swin = F.interpolate(f_swin, size=f_conv.shape[2:], mode='bilinear', align_corners=False)
            
        f_conv_norm = self.ln_conv(f_conv.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        f_swin_norm = self.ln_swin(f_swin.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        f_conv_proj = self.proj_conv(f_conv_norm) * self.alpha
        f_swin_proj = self.proj_swin(f_swin_norm) * self.beta
        
        f_cal = torch.cat([f_conv_proj, f_swin_proj], dim=1)
        
        a_s = self.spatial_gate(f_cal)
        f_spatial = f_cal * a_s
        
        n, c, h, w = f_cal.shape
        z = F.adaptive_avg_pool2d(f_cal, (1, 1)).flatten(1)
        a_c = self.channel_mlp(z).view(n, c, 1, 1)
        f_channel = f_cal * a_c
        
        return f_spatial + f_channel

class DentalLens(nn.Module):
    def __init__(self, num_classes=6):
        super(DentalLens, self).__init__()
        
        base_convnext = models.convnext_tiny(weights=None) 
        self.branch1 = create_feature_extractor(base_convnext, return_nodes={'features': 'out'})
        
        base_swin = models.swin_t(weights=None)
        self.branch2 = create_feature_extractor(base_swin, return_nodes={'features': 'out'})

        self.agff = AGFFBlock(in_channels=768)
        
        self.final_ln = nn.LayerNorm(768)
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(768, num_classes)
        
    def forward(self, x):
        f_conv = self.branch1(x)['out']
        f_swin = self.branch2(x)['out']
        f_swin = f_swin.permute(0, 3, 1, 2)
        
        f_fused = self.agff(f_conv, f_swin)
        
        f_perm = f_fused.permute(0, 2, 3, 1)
        f_norm = self.final_ln(f_perm).permute(0, 3, 1, 2)
        v = F.adaptive_avg_pool2d(f_norm, (1, 1)).flatten(1)
        
        v = self.dropout(v)
        logits = self.classifier(v)
        
        return logits

# ==================== PREDICTOR LOGIC ====================
class LIMEPredictor:
    """LIME-enabled predictor for DentalLens V6"""
    
    def __init__(self):
        self.device = DEVICE
        self.classes = CLASSES
        
        # Exact transforms from your testing script
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        logger.info("Loading DentalLens V6 model...")
        self.model = DentalLens(num_classes=NUM_CLASSES)
        
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=self.device)
            # Handle standard state_dict or wrapped dict
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        except Exception as e:
            logger.error(f"Failed to load model weights: {e}")
            raise
            
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Predictor initialized on {self.device}")
    
    def predict(self, image_bytes):
        """Quick prediction without LIME"""
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(img_tensor)
                probs = F.softmax(output, dim=1)[0].cpu().numpy()
                
            pred_idx = np.argmax(probs)
            pred_class = self.classes[pred_idx]
            confidence = probs[pred_idx]
            
            return {
                'prediction': pred_class,
                'confidence': float(confidence),
                'all_probabilities': {
                    self.classes[i]: float(probs[i]) for i in range(len(self.classes))
                }
            }
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise
    
    def predict_with_lime(self, image_bytes, num_samples=100):
        """Prediction with LIME explanation"""
        try:
            logger.info(f"Generating LIME explanation with {num_samples} samples...")
            
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            # Ensure resizing matches the model dimensions before converting to array for LIME
            image = image.resize(IMAGE_SIZE)
            image_array = np.array(image)
            
            prediction_result = self.predict(image_bytes)
            predicted_class_name = prediction_result['prediction']
            predicted_class_idx = self.classes.index(predicted_class_name)
            
            def predict_fn(images):
                batch_tensors = []
                for img in images:
                    pil_img = Image.fromarray(img.astype(np.uint8))
                    img_tensor = self.transform(pil_img)
                    batch_tensors.append(img_tensor)
                
                batch_tensors = torch.stack(batch_tensors).to(self.device)
                
                self.model.eval()
                all_probs = []
                # Process in batches to avoid OOM
                batch_size = 32
                with torch.no_grad():
                    for i in range(0, len(batch_tensors), batch_size):
                        batch = batch_tensors[i:i+batch_size]
                        outputs = self.model(batch)
                        probs = F.softmax(outputs, dim=1).cpu().numpy()
                        all_probs.append(probs)
                        
                return np.vstack(all_probs)
            
            explainer = lime_image.LimeImageExplainer(random_state=42)
            explanation = explainer.explain_instance(
                image_array,
                predict_fn,
                top_labels=NUM_CLASSES,
                hide_color=0,
                num_samples=num_samples,
                segmentation_fn=lambda x: slic(x, n_segments=50, compactness=10, sigma=1, start_label=0),
                random_seed=42
            )
            
            logger.info("Creating comprehensive LIME visualization...")
            
            temp, mask = explanation.get_image_and_mask(
                predicted_class_idx,
                positive_only=False,
                num_features=10,
                hide_rest=False
            )
            
            temp_positive, mask_positive = explanation.get_image_and_mask(
                predicted_class_idx,
                positive_only=True,
                num_features=5,
                hide_rest=False
            )
            
            temp_negative, mask_negative = explanation.get_image_and_mask(
                predicted_class_idx,
                positive_only=False,
                num_features=5,
                hide_rest=False,
                negative_only=True
            )
            
            fig, axes = plt.subplots(2, 4, figsize=(20, 12))
            fig.suptitle(f'LIME Explanation for {predicted_class_name}', fontsize=16, fontweight='bold')
            
            axes[0, 0].imshow(image)
            axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
            axes[0, 0].axis('off')
            
            segments = explanation.segments
            segmented_img = mark_boundaries(image_array/255.0, segments)
            axes[0, 1].imshow(segmented_img)
            axes[0, 1].set_title(f'Superpixel Segmentation\n({len(np.unique(segments))} segments)', 
                               fontsize=12, fontweight='bold')
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(mark_boundaries(temp/255.0, mask))
            axes[0, 2].set_title('Complete Explanation\n(Green=Support, Red=Against)', 
                               fontsize=12, fontweight='bold')
            axes[0, 2].axis('off')
            
            axes[0, 3].imshow(mark_boundaries(temp_positive/255.0, mask_positive))
            axes[0, 3].set_title('Positive Evidence\n(Supports Diagnosis)', 
                               fontsize=12, fontweight='bold')
            axes[0, 3].axis('off')
            
            axes[1, 0].imshow(mark_boundaries(temp_negative/255.0, mask_negative))
            axes[1, 0].set_title('Negative Evidence\n(Against Diagnosis)', 
                               fontsize=12, fontweight='bold')
            axes[1, 0].axis('off')
            
            importance_map = self._create_importance_heatmap(explanation, predicted_class_idx)
            im = axes[1, 1].imshow(importance_map, cmap='RdYlBu_r', alpha=0.8)
            axes[1, 1].imshow(image_array, alpha=0.5)
            axes[1, 1].set_title('Importance Heatmap\n(Warmer = More Important)', 
                               fontsize=12, fontweight='bold')
            axes[1, 1].axis('off')
            plt.colorbar(im, ax=axes[1, 1], fraction=0.046)
            
            self._plot_top_regions(axes[1, 2], explanation, image_array, predicted_class_idx)
            self._plot_quantitative_analysis(axes[1, 3], explanation, predicted_class_idx, predicted_class_name)
            
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            local_exp = explanation.local_exp[predicted_class_idx]
            positive_sum = sum(imp for _, imp in local_exp if imp > 0)
            negative_sum = sum(imp for _, imp in local_exp if imp < 0)
            net_evidence = positive_sum + negative_sum
            
            clinical_interpretation = []
            if positive_sum > abs(negative_sum):
                clinical_interpretation.append("The model found more supporting evidence than contradicting evidence")
                clinical_interpretation.append("Key pathological regions were identified and weighted appropriately")
            else:
                clinical_interpretation.append("The model found mixed or contradictory evidence")
                clinical_interpretation.append("Consider reviewing the diagnosis or obtaining additional images")
            
            logger.info("LIME explanation generated successfully")
            
            return {
                'explanation_image': img_base64,
                'prediction': prediction_result,
                'num_samples': num_samples,
                'lime_statistics': {
                    'total_positive_evidence': float(positive_sum),
                    'total_negative_evidence': float(negative_sum),
                    'net_evidence': float(net_evidence),
                    'clinical_interpretation': clinical_interpretation
                }
            }
            
        except Exception as e:
            logger.error(f"LIME explanation error: {str(e)}")
            raise

    def _create_importance_heatmap(self, explanation, predicted_class_idx):
        segments = explanation.segments
        importance_map = np.zeros_like(segments, dtype=float)
        local_exp = explanation.local_exp[predicted_class_idx]
        for segment_id, importance in local_exp:
            importance_map[segments == segment_id] = importance
        return importance_map
    
    def _plot_top_regions(self, ax, explanation, image_array, predicted_class_idx):
        local_exp = explanation.local_exp[predicted_class_idx]
        top_regions = sorted(local_exp, key=lambda x: abs(x[1]), reverse=True)[:5]
        segments = explanation.segments
        top_regions_mask = np.zeros_like(segments)
        
        for i, (segment_id, importance) in enumerate(top_regions):
            top_regions_mask[segments == segment_id] = i + 1
            
        masked_image = image_array.copy()
        colored_mask = plt.cm.Set1(top_regions_mask / 5.0)
        
        for i in range(3):
            masked_image[:, :, i] = np.where(
                top_regions_mask > 0,
                0.6 * masked_image[:, :, i] + 0.4 * 255 * colored_mask[:, :, i],
                masked_image[:, :, i]
            )
            
        ax.imshow(masked_image.astype(np.uint8))
        ax.set_title('Top 5 Contributing Regions', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        legend_text = []
        for i, (segment_id, importance) in enumerate(top_regions):
            sign = "+" if importance > 0 else ""
            legend_text.append(f"Region {i+1}: {sign}{importance:.3f}")
            
        ax.text(1.02, 1, '\n'.join(legend_text), transform=ax.transAxes, 
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    def _plot_quantitative_analysis(self, ax, explanation, predicted_class_idx, disease_name):
        local_exp = explanation.local_exp[predicted_class_idx]
        positive_contrib = [imp for _, imp in local_exp if imp > 0]
        negative_contrib = [imp for _, imp in local_exp if imp < 0]
        
        stats_text = f"Quantitative Analysis\n" + "="*25 + "\n"
        stats_text += f"Disease: {disease_name}\n\n"
        stats_text += f"Total Regions: {len(local_exp)}\n"
        stats_text += f"Supporting: {len(positive_contrib)}\n"
        stats_text += f"Against: {len(negative_contrib)}\n\n"
        
        if positive_contrib:
            stats_text += f"Positive Evidence:\n"
            stats_text += f"• Mean: {np.mean(positive_contrib):.4f}\n"
            stats_text += f"• Max: {np.max(positive_contrib):.4f}\n"
            stats_text += f"• Sum: {np.sum(positive_contrib):.4f}\n\n"
            
        if negative_contrib:
            stats_text += f"Negative Evidence:\n"
            stats_text += f"• Mean: {np.abs(np.mean(negative_contrib)):.4f}\n"
            stats_text += f"• Min: {np.abs(np.min(negative_contrib)):.4f}\n"
            stats_text += f"• Sum: {np.abs(np.sum(negative_contrib)):.4f}\n\n"
            
        net_support = np.sum(positive_contrib) - np.abs(np.sum(negative_contrib))
        stats_text += f"Net Support: {net_support:.4f}\n"
        
        if net_support > 0.1:
            assessment = "Strong Support"
        elif net_support > 0.05:
            assessment = "Moderate Support"
        elif net_support > -0.05:
            assessment = "Weak/Mixed Evidence"
        else:
            assessment = "Contradictory Evidence"
            
        stats_text += f"Assessment: {assessment}"
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=10, fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        ax.set_title('Statistical Summary', fontsize=12, fontweight='bold')
        ax.axis('off')

# Global instance
_lime_predictor = None

def get_lime_predictor():
    global _lime_predictor
    if _lime_predictor is None:
        _lime_predictor = LIMEPredictor()
    return _lime_predictor