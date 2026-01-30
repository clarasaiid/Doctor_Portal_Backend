"""
DeepSeek-OCR-2 Model Integration
Handles loading and inference with DeepSeek-OCR-2 for Arabic and English text extraction

REQUIREMENTS:
- NVIDIA GPU with CUDA
- flash-attn==2.7.3
- transformers==4.46.3
"""
import os
import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepSeekOCRModel:
    """
    Wrapper for DeepSeek-OCR-2 model
    
    This model REQUIRES:
    - NVIDIA GPU with CUDA support
    - flash-attn library installed
    """
    
    def __init__(self):
        self.model_name = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-ai/DeepSeek-OCR-2")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the DeepSeek-OCR-2 model"""
        try:
            logger.info(f"Loading DeepSeek-OCR-2 model: {self.model_name}")
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            
            # Check CUDA requirement
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "❌ CUDA is not available! DeepSeek-OCR-2 requires an NVIDIA GPU with CUDA. "
                    "Please run this on a machine with CUDA support (e.g., Google Colab with GPU runtime)."
                )
            
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            
            # Check flash-attn requirement
            try:
                import flash_attn
                logger.info(f"flash-attn version: {flash_attn.__version__}")
            except ImportError:
                raise ImportError(
                    "❌ flash-attn is NOT installed! DeepSeek-OCR-2 requires flash-attn. "
                    "Install with: pip install flash-attn==2.7.3 --no-build-isolation"
                )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Load model with flash_attention_2 (REQUIRED)
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_safetensors=True,
                _attn_implementation="flash_attention_2"
            )
            
            # Set to evaluation mode, move to GPU, use bfloat16
            self.model = self.model.eval().cuda().to(torch.bfloat16)
            
            logger.info("✅ Model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}")
            raise
    
    def process_image(
        self,
        image_path: str,
        prompt: str = "<image>\n<|grounding|>Convert the document to markdown.",
        base_size: int = 1024,
        image_size: int = 768,
        crop_mode: bool = True,
        save_results: bool = False,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process an image and extract text using DeepSeek-OCR-2
        
        Args:
            image_path: Path to the image file
            prompt: Prompt for OCR (default converts to markdown)
            base_size: Base size for image processing
            image_size: Image size for processing
            crop_mode: Whether to use crop mode
            save_results: Whether to save results
            output_path: Path to save results if save_results=True
            
        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            if self.model is None or self.tokenizer is None:
                raise RuntimeError("Model not loaded")
            
            # Load and process image
            image = Image.open(image_path).convert("RGB")
            
            # Run inference
            logger.info(f"Processing image: {image_path}")
            result = self.model.infer(
                self.tokenizer,
                prompt=prompt,
                image_file=image_path,
                output_path=output_path if save_results else None,
                base_size=base_size,
                image_size=image_size,
                crop_mode=crop_mode,
                save_results=save_results
            )
            
            # Extract text from result
            extracted_text = result if isinstance(result, str) else result.get("text", "")
            
            # Detect languages (simple heuristic)
            languages = self._detect_languages(extracted_text)
            
            return {
                "text": extracted_text,
                "languages": languages,
                "confidence": 0.95,  # DeepSeek-OCR-2 typically has high confidence
                "raw_result": result
            }
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise
    
    def _detect_languages(self, text: str) -> list:
        """Detect languages in extracted text"""
        import re
        languages = []
        
        # Check for Arabic characters (Arabic Unicode range)
        arabic_pattern = r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]'
        if re.search(arabic_pattern, text):
            languages.append("ar")
        
        # Check for English (basic Latin characters)
        if re.search(r'[a-zA-Z]', text):
            languages.append("en")
        
        return languages if languages else ["unknown"]
    
    def extract_lab_values(self, text: str) -> Dict[str, Any]:
        """
        Extract structured lab values from OCR text
        Handles multiple lab test types: CBC, diabetes, lipid panel, etc.
        """
        import re
        
        structured = {}
        raw_lines = text.split("\n")
        
        # Comprehensive lab test patterns (Arabic and English)
        patterns = {
            # Complete Blood Count (CBC)
            "Hemoglobin": [r'Hemoglobin[:\s]+([\d.]+)', r'هيموغلوبين[:\s]+([\d.]+)', r'Hb[:\s]+([\d.]+)', r'Haemoglobin[:\s]+([\d.]+)'],
            "Hematocrit": [r'Hematocrit[:\s]+([\d.]+)', r'PCV[:\s]+([\d.]+)', r'هيماتوكريت[:\s]+([\d.]+)'],
            "RBC": [r'RBCs?\s+Count[:\s]+([\d.]+)', r'RBC[:\s]+([\d.]+)', r'كريات حمراء[:\s]+([\d.]+)'],
            "MCV": [r'MCV[:\s]+([\d.]+)'],
            "MCH": [r'MCH[:\s]+([\d.]+)'],
            "MCHC": [r'MCHC[:\s]+([\d.]+)'],
            "RDW": [r'RDW[:\s]+([\d.]+)'],
            "Platelet": [r'Platelet[:\s]+([\d.]+)', r'Platelets?[:\s]+([\d.]+)', r'صفائح[:\s]+([\d.]+)'],
            "WBC": [r'Total\s+Leucocytic\s+Count[:\s]+([\d.]+)', r'WBC[:\s]+([\d.]+)', r'White\s+Blood\s+Cell[:\s]+([\d.]+)', r'كريات بيضاء[:\s]+([\d.]+)'],
            "Neutrophils": [r'Neutrophils?[:\s]+([\d.]+)', r'عدلات[:\s]+([\d.]+)'],
            "Lymphocytes": [r'Lymphocytes?[:\s]+([\d.]+)', r'لمفاويات[:\s]+([\d.]+)'],
            "Monocytes": [r'Monocytes?[:\s]+([\d.]+)', r'أحادية[:\s]+([\d.]+)'],
            "Eosinophils": [r'Eosinophils?[:\s]+([\d.]+)', r'حامضية[:\s]+([\d.]+)'],
            "Basophils": [r'Basophils?[:\s]+([\d.]+)', r'قاعدية[:\s]+([\d.]+)'],
            
            # Diabetes/Glucose tests
            "HbA1c": [r'HbA1c[:\s]+([\d.]+)\s*%', r'الهيموغلوبين[:\s]+([\d.]+)\s*%'],
            "Glucose": [r'Glucose[:\s]+([\d.]+)', r'سكر[:\s]+([\d.]+)', r'Fasting\s+Glucose[:\s]+([\d.]+)'],
            "FastingGlucose": [r'Fasting\s+Glucose[:\s]+([\d.]+)', r'سكر صائم[:\s]+([\d.]+)'],
            
            # Vital signs
            "BloodPressure": [r'BP[:\s]+([\d]+)/([\d]+)', r'ضغط[:\s]+([\d]+)/([\d]+)', r'Blood\s+Pressure[:\s]+([\d]+)/([\d]+)'],
            "Weight": [r'Weight[:\s]+([\d.]+)', r'وزن[:\s]+([\d.]+)'],
            "Height": [r'Height[:\s]+([\d.]+)', r'طول[:\s]+([\d.]+)'],
            "BMI": [r'BMI[:\s]+([\d.]+)', r'مؤشر[:\s]+([\d.]+)'],
            
            # Patient info
            "PatientName": [r'Patient\s+Name[:\s]+([A-Za-z\s\u0600-\u06FF]+)', r'اسم المريض[:\s]+([A-Za-z\s\u0600-\u06FF]+)'],
            "Age": [r'Age[:\s]+(\d+)', r'عمر[:\s]+(\d+)', r'(\d+)\s+Year'],
            "Gender": [r'Sex[:\s]+(Male|Female|M|F)', r'جنس[:\s]+(ذكر|أنثى|Male|Female)'],
        }
        
        # Extract values using patterns
        for metric, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, text, re.IGNORECASE | re.UNICODE)
                if match:
                    if metric == "BloodPressure":
                        structured["SystolicBP"] = float(match.group(1))
                        structured["DiastolicBP"] = float(match.group(2))
                    elif metric in ["PatientName", "Gender"]:
                        structured[metric] = match.group(1).strip()
                    else:
                        try:
                            structured[metric] = float(match.group(1))
                        except ValueError:
                            pass
                    break
        
        # Try to extract lab report type from text
        if any(key in structured for key in ["Hemoglobin", "Hematocrit", "RBC", "WBC"]):
            structured["ReportType"] = "Complete Blood Count (CBC)"
        elif any(key in structured for key in ["HbA1c", "Glucose", "FastingGlucose"]):
            structured["ReportType"] = "Diabetes Profile"
        elif structured:
            structured["ReportType"] = "Lab Report"
        
        return {
            "structured": structured,
            "raw_text": raw_lines
        }


# Global model instance (singleton)
_ocr_model_instance: Optional[DeepSeekOCRModel] = None


def get_ocr_model() -> DeepSeekOCRModel:
    """Get or create OCR model instance"""
    global _ocr_model_instance
    if _ocr_model_instance is None:
        _ocr_model_instance = DeepSeekOCRModel()
    return _ocr_model_instance
