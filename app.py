import os
import io
import hashlib
from datetime import datetime

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from dotenv import load_dotenv
from groq import Groq
from gtts import gTTS
from audio_recorder_streamlit import audio_recorder

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as PDFImage,
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

from groq_assistant import explain_screening_result


# =========================================================
# 1. CONFIGURATION
# =========================================================

load_dotenv()

st.set_page_config(
    page_title="Rural Eye Screening AI",
    page_icon="👁️",
    layout="wide",
)

MODEL_PATH = os.path.join(
    "models",
    "eye_classifier.pth",
)


CLASS_NAMES = [
    "cataract",
    "normal",
    "pterygium",
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

groq_client = (
    Groq(api_key=GROQ_API_KEY)
    if GROQ_API_KEY
    else None
)


# =========================================================
# 2. LANGUAGE CONFIGURATION
# =========================================================

LANGUAGES = {
    "English": {
        "stt": "en",
        "tts": "en",
        "diseases": {
            "cataract": "Cataract",
            "normal": "Normal",
            "pterygium": "Pterygium",
        },
    },
    "Hindi (हिंदी)": {
        "stt": "hi",
        "tts": "hi",
        "diseases": {
            "cataract": "मोतियाबिंद",
            "normal": "सामान्य",
            "pterygium": "प्टेरिजियम",
        },
    },
    "Punjabi (ਪੰਜਾਬੀ)": {
        "stt": "pa",
        "tts": "pa",
        "diseases": {
            "cataract": "ਮੋਤੀਆਬਿੰਦ",
            "normal": "ਸਧਾਰਨ",
            "pterygium": "ਪਟੇਰੀਜੀਅਮ",
        },
    },
    "Urdu (اردو)": {
        "stt": "ur",
        "tts": "ur",
        "diseases": {
            "cataract": "موتیا بند",
            "normal": "معمول کے مطابق",
            "pterygium": "پٹریجیم",
        },
    },
    "Kannada (ಕನ್ನಡ)": {
        "stt": "kn",
        "tts": "kn",
        "diseases": {
            "cataract": "ಕಣ್ಣಿನ ಪೊರೆ",
            "normal": "ಸಾಮಾನ್ಯ",
            "pterygium": "ಪ್ಟೆರಿಜಿಯಮ್",
        },
    },
    "Tamil (தமிழ்)": {
        "stt": "ta",
        "tts": "ta",
        "diseases": {
            "cataract": "கண்புரை",
            "normal": "இயல்பானது",
            "pterygium": "ப்டெரிஜியம்",
        },
    },
    "Telugu (తెలుగు)": {
        "stt": "te",
        "tts": "te",
        "diseases": {
            "cataract": "కంటిశుక్లం",
            "normal": "సాధారణం",
            "pterygium": "ప్టెరిజియం",
        },
    },
}


# =========================================================
# 3. TRANSLATED USER INTERFACE
# =========================================================

UI_TEXT = {
    "English": {
        "title": "Rural Eye Screening AI Assistant",
        "subtitle": (
            "AI-assisted eye screening with multilingual "
            "support and telemedicine guidance."
        ),
        "language": "Choose your language",
        "about": "About this system",
        "about_text": (
            "Upload an eye image to receive an AI screening result. "
            "This prototype classifies cataract, normal eyes, and "
            "pterygium. It does not currently classify pinguecula."
        ),
        "upload": "Upload an eye image",
        "upload_help": "Upload a clear JPG, JPEG, or PNG eye image.",
        "screen": "Analyze eye image",
        "processing": "Analyzing image...",
        "result": "Screening result",
        "confidence": "Model score",
        "explanation": "AI explanation",
        "generate": "Generate explanation",
        "generating": "Preparing explanation...",
        "voice_title": "Voice Assistant",
        "voice_help": (
            "Record a question about your screening result. "
            "The assistant will answer in your selected language."
        ),
        "record": "Click to record your question",
        "transcribing": "Understanding your voice...",
        "answering": "Preparing your answer...",
        "you_asked": "You asked",
        "voice_answer": "Assistant's answer",
        "pdf_title": "Download screening report",
        "pdf_button": "Download PDF report",
        "disclaimer": "Important medical notice",
        "disclaimer_text": (
            "This is an experimental AI screening tool, not a "
            "confirmed diagnosis. Consult a qualified eye-care "
            "professional. Seek urgent care for sudden vision loss, "
            "severe eye pain, or serious eye injury."
        ),
        "upload_first": "Upload an eye image to begin.",
        "api_missing": (
            "GROQ_API_KEY is missing. Add it to your .env file "
            "to enable AI explanations and voice features."
        ),
        "footer": "Academic prototype — not a substitute for clinical care.",
        "error": "An error occurred",
        "invalid": "Please upload a valid eye image.",
        "confidence_note": (
            "The model score is not a calibrated clinical probability."
        ),
        "new_image": "A new image was uploaded. Run screening again.",
        "language_changed": (
            "Your language changed. Generate the explanation again "
            "to receive it in the selected language."
        ),
        "no_result": (
            "Run the screening first to use the voice assistant."
        ),
    },
    "Hindi (हिंदी)": {
        "title": "ग्रामीण नेत्र जाँच AI सहायक",
        "subtitle": "बहुभाषी सहायता और टेलीमेडिसिन मार्गदर्शन के साथ AI नेत्र जाँच।",
        "language": "अपनी भाषा चुनें",
        "about": "इस प्रणाली के बारे में",
        "about_text": "आँख की तस्वीर अपलोड करके AI जाँच परिणाम प्राप्त करें। यह प्रणाली मोतियाबिंद, सामान्य आँख और प्टेरिजियम की पहचान करती है।",
        "upload": "आँख की तस्वीर अपलोड करें",
        "upload_help": "स्पष्ट JPG, JPEG या PNG तस्वीर अपलोड करें।",
        "screen": "आँख की तस्वीर का विश्लेषण करें",
        "processing": "तस्वीर का विश्लेषण हो रहा है...",
        "result": "जाँच परिणाम",
        "confidence": "मॉडल स्कोर",
        "explanation": "AI द्वारा व्याख्या",
        "generate": "व्याख्या तैयार करें",
        "generating": "व्याख्या तैयार हो रही है...",
        "voice_title": "वॉइस सहायक",
        "voice_help": "अपनी जाँच के बारे में प्रश्न रिकॉर्ड करें।",
        "record": "प्रश्न रिकॉर्ड करने के लिए क्लिक करें",
        "transcribing": "आपकी आवाज़ समझी जा रही है...",
        "answering": "उत्तर तैयार हो रहा है...",
        "you_asked": "आपका प्रश्न",
        "voice_answer": "सहायक का उत्तर",
        "pdf_title": "जाँच रिपोर्ट डाउनलोड करें",
        "pdf_button": "PDF रिपोर्ट डाउनलोड करें",
        "disclaimer": "महत्वपूर्ण चिकित्सा सूचना",
        "disclaimer_text": "यह AI जाँच उपकरण है, निश्चित निदान नहीं। नेत्र विशेषज्ञ से सलाह लें। अचानक दृष्टि खोने, तेज दर्द या गंभीर चोट पर तुरंत चिकित्सा सहायता लें।",
        "upload_first": "शुरू करने के लिए आँख की तस्वीर अपलोड करें।",
        "api_missing": ".env फ़ाइल में GROQ_API_KEY जोड़ें।",
        "footer": "शैक्षणिक प्रोटोटाइप — चिकित्सा देखभाल का विकल्प नहीं।",
        "error": "त्रुटि हुई",
        "invalid": "कृपया सही आँख की तस्वीर अपलोड करें।",
        "confidence_note": "मॉडल स्कोर चिकित्सकीय रूप से कैलिब्रेट की गई संभावना नहीं है।",
        "new_image": "नई तस्वीर अपलोड हुई है। फिर से जाँच करें।",
        "language_changed": "चुनी गई भाषा में व्याख्या पाने के लिए उसे दोबारा तैयार करें।",
        "no_result": "वॉइस सहायक के लिए पहले जाँच करें।",
    },
    "Punjabi (ਪੰਜਾਬੀ)": {
        "title": "ਪੇਂਡੂ ਅੱਖਾਂ ਦੀ ਜਾਂਚ AI ਸਹਾਇਕ",
        "subtitle": "ਬਹੁਭਾਸ਼ੀ ਸਹਾਇਤਾ ਅਤੇ ਟੈਲੀਮੈਡੀਸਨ ਮਾਰਗਦਰਸ਼ਨ ਨਾਲ ਅੱਖਾਂ ਦੀ ਜਾਂਚ।",
        "language": "ਆਪਣੀ ਭਾਸ਼ਾ ਚੁਣੋ",
        "about": "ਇਸ ਪ੍ਰਣਾਲੀ ਬਾਰੇ",
        "about_text": "AI ਜਾਂਚ ਲਈ ਅੱਖ ਦੀ ਤਸਵੀਰ ਅੱਪਲੋਡ ਕਰੋ। ਇਹ ਮੋਤੀਆਬਿੰਦ, ਆਮ ਅੱਖ ਅਤੇ ਪਟੇਰੀਜੀਅਮ ਦੀ ਵਰਗੀਕਰਨ ਕਰਦੀ ਹੈ।",
        "upload": "ਅੱਖ ਦੀ ਤਸਵੀਰ ਅੱਪਲੋਡ ਕਰੋ",
        "upload_help": "ਸਾਫ਼ JPG, JPEG ਜਾਂ PNG ਤਸਵੀਰ ਅੱਪਲੋਡ ਕਰੋ।",
        "screen": "ਤਸਵੀਰ ਦੀ ਜਾਂਚ ਕਰੋ",
        "processing": "ਤਸਵੀਰ ਦੀ ਜਾਂਚ ਹੋ ਰਹੀ ਹੈ...",
        "result": "ਜਾਂਚ ਦਾ ਨਤੀਜਾ",
        "confidence": "ਮਾਡਲ ਸਕੋਰ",
        "explanation": "AI ਵਿਆਖਿਆ",
        "generate": "ਵਿਆਖਿਆ ਬਣਾਓ",
        "generating": "ਵਿਆਖਿਆ ਤਿਆਰ ਹੋ ਰਹੀ ਹੈ...",
        "voice_title": "ਆਵਾਜ਼ ਸਹਾਇਕ",
        "voice_help": "ਆਪਣੀ ਜਾਂਚ ਬਾਰੇ ਸਵਾਲ ਰਿਕਾਰਡ ਕਰੋ।",
        "record": "ਸਵਾਲ ਰਿਕਾਰਡ ਕਰਨ ਲਈ ਕਲਿੱਕ ਕਰੋ",
        "transcribing": "ਤੁਹਾਡੀ ਆਵਾਜ਼ ਸਮਝੀ ਜਾ ਰਹੀ ਹੈ...",
        "answering": "ਜਵਾਬ ਤਿਆਰ ਹੋ ਰਿਹਾ ਹੈ...",
        "you_asked": "ਤੁਹਾਡਾ ਸਵਾਲ",
        "voice_answer": "ਸਹਾਇਕ ਦਾ ਜਵਾਬ",
        "pdf_title": "ਜਾਂਚ ਰਿਪੋਰਟ ਡਾਊਨਲੋਡ ਕਰੋ",
        "pdf_button": "PDF ਰਿਪੋਰਟ ਡਾਊਨਲੋਡ ਕਰੋ",
        "disclaimer": "ਮਹੱਤਵਪੂਰਨ ਡਾਕਟਰੀ ਜਾਣਕਾਰੀ",
        "disclaimer_text": "ਇਹ AI ਜਾਂਚ ਸਾਧਨ ਹੈ, ਪੱਕਾ ਨਿਦਾਨ ਨਹੀਂ। ਅੱਖਾਂ ਦੇ ਮਾਹਿਰ ਨਾਲ ਸਲਾਹ ਕਰੋ। ਅਚਾਨਕ ਨਜ਼ਰ ਘਟਣ, ਤੇਜ਼ ਦਰਦ ਜਾਂ ਗੰਭੀਰ ਸੱਟ ਲਈ ਤੁਰੰਤ ਡਾਕਟਰੀ ਮਦਦ ਲਵੋ।",
        "upload_first": "ਸ਼ੁਰੂ ਕਰਨ ਲਈ ਅੱਖ ਦੀ ਤਸਵੀਰ ਅੱਪਲੋਡ ਕਰੋ।",
        "api_missing": ".env ਫਾਈਲ ਵਿੱਚ GROQ_API_KEY ਜੋੜੋ।",
        "footer": "ਅਕਾਦਮਿਕ ਪ੍ਰੋਟੋਟਾਈਪ — ਡਾਕਟਰੀ ਦੇਖਭਾਲ ਦਾ ਬਦਲ ਨਹੀਂ।",
        "error": "ਗਲਤੀ ਹੋਈ",
        "invalid": "ਕਿਰਪਾ ਕਰਕੇ ਸਹੀ ਤਸਵੀਰ ਅੱਪਲੋਡ ਕਰੋ।",
        "confidence_note": "ਮਾਡਲ ਸਕੋਰ ਡਾਕਟਰੀ ਤੌਰ 'ਤੇ ਕੈਲੀਬਰੇਟ ਕੀਤੀ ਸੰਭਾਵਨਾ ਨਹੀਂ ਹੈ।",
        "new_image": "ਨਵੀਂ ਤਸਵੀਰ ਅੱਪਲੋਡ ਹੋਈ ਹੈ। ਮੁੜ ਜਾਂਚ ਕਰੋ।",
        "language_changed": "ਚੁਣੀ ਭਾਸ਼ਾ ਵਿੱਚ ਵਿਆਖਿਆ ਲਈ ਇਸਨੂੰ ਮੁੜ ਬਣਾਓ।",
        "no_result": "ਆਵਾਜ਼ ਸਹਾਇਕ ਲਈ ਪਹਿਲਾਂ ਜਾਂਚ ਕਰੋ।",
    },
    "Urdu (اردو)": {
        "title": "دیہی آنکھوں کی جانچ کا AI معاون",
        "subtitle": "کثیر لسانی مدد اور ٹیلی میڈیسن رہنمائی کے ساتھ آنکھوں کی جانچ۔",
        "language": "اپنی زبان منتخب کریں",
        "about": "اس نظام کے بارے میں",
        "about_text": "AI جانچ کے لیے آنکھ کی تصویر اپ لوڈ کریں۔ یہ موتیا، عام آنکھ اور پٹریجیم کی درجہ بندی کرتا ہے۔",
        "upload": "آنکھ کی تصویر اپ لوڈ کریں",
        "upload_help": "واضح JPG، JPEG یا PNG تصویر اپ لوڈ کریں۔",
        "screen": "تصویر کا تجزیہ کریں",
        "processing": "تصویر کا تجزیہ جاری ہے...",
        "result": "جانچ کا نتیجہ",
        "confidence": "ماڈل اسکور",
        "explanation": "AI کی وضاحت",
        "generate": "وضاحت تیار کریں",
        "generating": "وضاحت تیار ہو رہی ہے...",
        "voice_title": "آواز کا معاون",
        "voice_help": "اپنی جانچ کے بارے میں سوال ریکارڈ کریں۔",
        "record": "سوال ریکارڈ کرنے کے لیے کلک کریں",
        "transcribing": "آپ کی آواز سمجھی جا رہی ہے...",
        "answering": "جواب تیار ہو رہا ہے...",
        "you_asked": "آپ کا سوال",
        "voice_answer": "معاون کا جواب",
        "pdf_title": "جانچ کی رپورٹ ڈاؤن لوڈ کریں",
        "pdf_button": "PDF رپورٹ ڈاؤن لوڈ کریں",
        "disclaimer": "اہم طبی اطلاع",
        "disclaimer_text": "یہ AI جانچ کا آلہ ہے، حتمی تشخیص نہیں۔ آنکھوں کے ماہر سے مشورہ کریں۔ اچانک نظر ختم ہونے، شدید درد یا سنگین چوٹ پر فوری طبی مدد لیں۔",
        "upload_first": "شروع کرنے کے لیے آنکھ کی تصویر اپ لوڈ کریں۔",
        "api_missing": ".env فائل میں GROQ_API_KEY شامل کریں۔",
        "footer": "تعلیمی پروٹوٹائپ — طبی نگہداشت کا متبادل نہیں۔",
        "error": "خرابی پیش آئی",
        "invalid": "براہ کرم درست تصویر اپ لوڈ کریں۔",
        "confidence_note": "ماڈل اسکور طبی طور پر کیلیبریٹ شدہ امکان نہیں ہے۔",
        "new_image": "نئی تصویر اپ لوڈ ہوئی ہے۔ دوبارہ جانچ کریں۔",
        "language_changed": "منتخب زبان میں وضاحت حاصل کرنے کے لیے اسے دوبارہ تیار کریں۔",
        "no_result": "آواز کے معاون کے لیے پہلے جانچ کریں۔",
    },
    "Kannada (ಕನ್ನಡ)": {
        "title": "ಗ್ರಾಮೀಣ ಕಣ್ಣಿನ ತಪಾಸಣೆ AI ಸಹಾಯಕ",
        "subtitle": "ಬಹುಭಾಷಾ ಬೆಂಬಲ ಮತ್ತು ಟೆಲಿಮೆಡಿಸಿನ್ ಮಾರ್ಗದರ್ಶನದೊಂದಿಗೆ AI ಕಣ್ಣಿನ ತಪಾಸಣೆ.",
        "language": "ನಿಮ್ಮ ಭಾಷೆಯನ್ನು ಆಯ್ಕೆಮಾಡಿ",
        "about": "ಈ ವ್ಯವಸ್ಥೆಯ ಬಗ್ಗೆ",
        "about_text": "AI ತಪಾಸಣೆಗಾಗಿ ಕಣ್ಣಿನ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ. ಈ ವ್ಯವಸ್ಥೆ ಕಣ್ಣಿನ ಪೊರೆ, ಸಾಮಾನ್ಯ ಕಣ್ಣು ಮತ್ತು ಪ್ಟೆರಿಜಿಯಮ್ ಅನ್ನು ವರ್ಗೀಕರಿಸುತ್ತದೆ.",
        "upload": "ಕಣ್ಣಿನ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
        "upload_help": "ಸ್ಪಷ್ಟವಾದ JPG, JPEG ಅಥವಾ PNG ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ.",
        "screen": "ಚಿತ್ರವನ್ನು ವಿಶ್ಲೇಷಿಸಿ",
        "processing": "ಚಿತ್ರವನ್ನು ವಿಶ್ಲೇಷಿಸಲಾಗುತ್ತಿದೆ...",
        "result": "ತಪಾಸಣೆ ಫಲಿತಾಂಶ",
        "confidence": "ಮಾದರಿ ಸ್ಕೋರ್",
        "explanation": "AI ವಿವರಣೆ",
        "generate": "ವಿವರಣೆ ರಚಿಸಿ",
        "generating": "ವಿವರಣೆ ಸಿದ್ಧವಾಗುತ್ತಿದೆ...",
        "voice_title": "ಧ್ವನಿ ಸಹಾಯಕ",
        "voice_help": "ನಿಮ್ಮ ತಪಾಸಣೆಯ ಬಗ್ಗೆ ಪ್ರಶ್ನೆಯನ್ನು ದಾಖಲಿಸಿ.",
        "record": "ಪ್ರಶ್ನೆ ದಾಖಲಿಸಲು ಕ್ಲಿಕ್ ಮಾಡಿ",
        "transcribing": "ನಿಮ್ಮ ಧ್ವನಿಯನ್ನು ಅರ್ಥಮಾಡಿಕೊಳ್ಳಲಾಗುತ್ತಿದೆ...",
        "answering": "ಉತ್ತರ ಸಿದ್ಧವಾಗುತ್ತಿದೆ...",
        "you_asked": "ನಿಮ್ಮ ಪ್ರಶ್ನೆ",
        "voice_answer": "ಸಹಾಯಕನ ಉತ್ತರ",
        "pdf_title": "ತಪಾಸಣೆ ವರದಿಯನ್ನು ಡೌನ್‌ಲೋಡ್ ಮಾಡಿ",
        "pdf_button": "PDF ವರದಿ ಡೌನ್‌ಲೋಡ್ ಮಾಡಿ",
        "disclaimer": "ಪ್ರಮುಖ ವೈದ್ಯಕೀಯ ಸೂಚನೆ",
        "disclaimer_text": "ಇದು AI ತಪಾಸಣಾ ಸಾಧನ ಮಾತ್ರ; ಖಚಿತ ರೋಗನಿರ್ಣಯವಲ್ಲ. ಕಣ್ಣಿನ ತಜ್ಞರನ್ನು ಸಂಪರ್ಕಿಸಿ. ಹಠಾತ್ ದೃಷ್ಟಿ ಕಳೆದುಕೊಳ್ಳುವುದು, ತೀವ್ರ ನೋವು ಅಥವಾ ಗಂಭೀರ ಗಾಯವಾದರೆ ತುರ್ತು ವೈದ್ಯಕೀಯ ಸಹಾಯ ಪಡೆಯಿರಿ.",
        "upload_first": "ಪ್ರಾರಂಭಿಸಲು ಕಣ್ಣಿನ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ.",
        "api_missing": ".env ಫೈಲ್‌ನಲ್ಲಿ GROQ_API_KEY ಸೇರಿಸಿ.",
        "footer": "ಶೈಕ್ಷಣಿಕ ಪ್ರೋಟೋಟೈಪ್ — ವೈದ್ಯಕೀಯ ಆರೈಕೆಗೆ ಪರ್ಯಾಯವಲ್ಲ.",
        "error": "ದೋಷ ಸಂಭವಿಸಿದೆ",
        "invalid": "ದಯವಿಟ್ಟು ಸರಿಯಾದ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ.",
        "confidence_note": "ಮಾದರಿ ಸ್ಕೋರ್ ವೈದ್ಯಕೀಯವಾಗಿ ಕ್ಯಾಲಿಬ್ರೇಟ್ ಮಾಡಿದ ಸಂಭವನೀಯತೆ ಅಲ್ಲ.",
        "new_image": "ಹೊಸ ಚಿತ್ರ ಅಪ್‌ಲೋಡ್ ಆಗಿದೆ. ಮತ್ತೆ ತಪಾಸಣೆ ಮಾಡಿ.",
        "language_changed": "ಆಯ್ಕೆ ಮಾಡಿದ ಭಾಷೆಯಲ್ಲಿ ವಿವರಣೆಯನ್ನು ಪಡೆಯಲು ಮತ್ತೆ ರಚಿಸಿ.",
        "no_result": "ಧ್ವನಿ ಸಹಾಯಕವನ್ನು ಬಳಸುವ ಮೊದಲು ತಪಾಸಣೆ ಮಾಡಿ.",
    },
    "Tamil (தமிழ்)": {
        "title": "கிராமப்புற கண் பரிசோதனை AI உதவியாளர்",
        "subtitle": "பலமொழி ஆதரவு மற்றும் தொலைமருத்துவ வழிகாட்டுதலுடன் AI கண் பரிசோதனை.",
        "language": "உங்கள் மொழியைத் தேர்ந்தெடுக்கவும்",
        "about": "இந்த அமைப்பைப் பற்றி",
        "about_text": "AI பரிசோதனைக்காக கண் படத்தைப் பதிவேற்றவும். இந்த அமைப்பு கண்புரை, இயல்பான கண் மற்றும் ப்டெரிஜியம் ஆகியவற்றை வகைப்படுத்துகிறது.",
        "upload": "கண் படத்தைப் பதிவேற்றவும்",
        "upload_help": "தெளிவான JPG, JPEG அல்லது PNG படத்தைப் பதிவேற்றவும்.",
        "screen": "கண் படத்தைப் பகுப்பாய்வு செய்யவும்",
        "processing": "படம் பகுப்பாய்வு செய்யப்படுகிறது...",
        "result": "பரிசோதனை முடிவு",
        "confidence": "மாதிரி மதிப்பெண்",
        "explanation": "AI விளக்கம்",
        "generate": "விளக்கத்தை உருவாக்கவும்",
        "generating": "விளக்கம் தயாராகிறது...",
        "voice_title": "குரல் உதவியாளர்",
        "voice_help": "உங்கள் பரிசோதனை பற்றி ஒரு கேள்வியைப் பதிவு செய்யவும்.",
        "record": "கேள்வியைப் பதிவு செய்ய கிளிக் செய்யவும்",
        "transcribing": "உங்கள் குரல் புரிந்துகொள்ளப்படுகிறது...",
        "answering": "பதில் தயாராகிறது...",
        "you_asked": "உங்கள் கேள்வி",
        "voice_answer": "உதவியாளரின் பதில்",
        "pdf_title": "பரிசோதனை அறிக்கையைப் பதிவிறக்கவும்",
        "pdf_button": "PDF அறிக்கையைப் பதிவிறக்கவும்",
        "disclaimer": "முக்கிய மருத்துவ அறிவிப்பு",
        "disclaimer_text": "இது AI பரிசோதனை கருவி மட்டுமே; உறுதியான நோயறிதல் அல்ல. கண் மருத்துவரை அணுகவும். திடீர் பார்வை இழப்பு, கடுமையான வலி அல்லது தீவிர காயம் ஏற்பட்டால் உடனடி மருத்துவ உதவி பெறவும்.",
        "upload_first": "தொடங்க கண் படத்தைப் பதிவேற்றவும்.",
        "api_missing": ".env கோப்பில் GROQ_API_KEY சேர்க்கவும்.",
        "footer": "கல்வி முன்மாதிரி — மருத்துவப் பராமரிப்புக்கு மாற்றாகாது.",
        "error": "பிழை ஏற்பட்டது",
        "invalid": "சரியான படத்தைப் பதிவேற்றவும்.",
        "confidence_note": "மாதிரி மதிப்பெண் மருத்துவ ரீதியாக அளவீடு செய்யப்பட்ட நிகழ்தகவு அல்ல.",
        "new_image": "புதிய படம் பதிவேற்றப்பட்டது. மீண்டும் பரிசோதிக்கவும்.",
        "language_changed": "தேர்ந்தெடுத்த மொழியில் விளக்கத்தைப் பெற மீண்டும் உருவாக்கவும்.",
        "no_result": "குரல் உதவியாளரைப் பயன்படுத்த முதலில் பரிசோதிக்கவும்.",
    },
    "Telugu (తెలుగు)": {
        "title": "గ్రామీణ కంటి పరీక్ష AI సహాయకుడు",
        "subtitle": "బహుభాషా మద్దతు మరియు టెలిమెడిసిన్ మార్గదర్శకత్వంతో AI కంటి పరీక్ష.",
        "language": "మీ భాషను ఎంచుకోండి",
        "about": "ఈ వ్యవస్థ గురించి",
        "about_text": "AI పరీక్ష కోసం కంటి చిత్రాన్ని అప్‌లోడ్ చేయండి. ఈ వ్యవస్థ కంటిశుక్లం, సాధారణ కన్ను మరియు ప్టెరిజియంను వర్గీకరిస్తుంది.",
        "upload": "కంటి చిత్రాన్ని అప్‌లోడ్ చేయండి",
        "upload_help": "స్పష్టమైన JPG, JPEG లేదా PNG చిత్రాన్ని అప్‌లోడ్ చేయండి.",
        "screen": "కంటి చిత్రాన్ని విశ్లేషించండి",
        "processing": "చిత్రాన్ని విశ్లేషిస్తోంది...",
        "result": "పరీక్ష ఫలితం",
        "confidence": "మోడల్ స్కోర్",
        "explanation": "AI వివరణ",
        "generate": "వివరణను రూపొందించండి",
        "generating": "వివరణ సిద్ధమవుతోంది...",
        "voice_title": "వాయిస్ సహాయకుడు",
        "voice_help": "మీ పరీక్ష గురించి ప్రశ్నను రికార్డ్ చేయండి.",
        "record": "ప్రశ్నను రికార్డ్ చేయడానికి క్లిక్ చేయండి",
        "transcribing": "మీ స్వరాన్ని అర్థం చేసుకుంటోంది...",
        "answering": "సమాధానం సిద్ధమవుతోంది...",
        "you_asked": "మీ ప్రశ్న",
        "voice_answer": "సహాయకుడి సమాధానం",
        "pdf_title": "పరీక్ష నివేదికను డౌన్‌లోడ్ చేయండి",
        "pdf_button": "PDF నివేదికను డౌన్‌లోడ్ చేయండి",
        "disclaimer": "ముఖ్యమైన వైద్య సమాచారం",
        "disclaimer_text": "ఇది AI స్క్రీనింగ్ సాధనం మాత్రమే; నిర్ధారిత రోగనిర్ధారణ కాదు. కంటి వైద్యుడిని సంప్రదించండి. అకస్మాత్తుగా చూపు కోల్పోవడం, తీవ్రమైన నొప్పి లేదా తీవ్రమైన గాయం ఉంటే వెంటనే వైద్య సహాయం పొందండి.",
        "upload_first": "ప్రారంభించడానికి కంటి చిత్రాన్ని అప్‌లోడ్ చేయండి.",
        "api_missing": ".env ఫైల్‌లో GROQ_API_KEY చేర్చండి.",
        "footer": "విద్యా ప్రోటోటైప్ — వైద్య సంరక్షణకు ప్రత్యామ్నాయం కాదు.",
        "error": "లోపం ఏర్పడింది",
        "invalid": "దయచేసి సరైన చిత్రాన్ని అప్‌లోడ్ చేయండి.",
        "confidence_note": "మోడల్ స్కోర్ వైద్యపరంగా కేలిబ్రేట్ చేసిన సంభావ్యత కాదు.",
        "new_image": "కొత్త చిత్రం అప్‌లోడ్ అయింది. మళ్లీ పరీక్షించండి.",
        "language_changed": "ఎంచుకున్న భాషలో వివరణ పొందడానికి మళ్లీ రూపొందించండి.",
        "no_result": "వాయిస్ సహాయకుడిని ఉపయోగించే ముందు పరీక్షించండి.",
    },
}


# =========================================================
# 4. MODEL LOADING
# =========================================================

@st.cache_resource
def load_eye_model(model_path):
    """Load the architecture that matches the supplied checkpoint.

    The previous version always created MobileNetV3-Small. The checkpoint
    currently producing the reported error has ShuffleNetV2 parameter names
    (for example, conv1/stage2/stage3/fc), so it must be loaded into
    ShuffleNetV2 instead.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}. "
            "Check MODEL_PATH and place the trained .pth file in that location."
        )

    try:
        checkpoint = torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=True,
        )
    except TypeError:
        # Compatibility with older PyTorch versions that do not accept
        # the weights_only argument.
        checkpoint = torch.load(model_path, map_location=DEVICE)

    saved_classes = None
    if isinstance(checkpoint, dict):
        saved_classes = checkpoint.get("class_names")
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise ValueError(
            "The checkpoint does not contain a valid model state dictionary."
        )

    # Remove DataParallel's prefix if the model was saved using multiple GPUs.
    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }

    if saved_classes is not None and list(saved_classes) != CLASS_NAMES:
        raise ValueError(
            "The checkpoint class order does not match app.py. "
            f"Checkpoint classes: {list(saved_classes)}; "
            f"app.py classes: {CLASS_NAMES}. Update CLASS_NAMES to the exact "
            "class order used when training the checkpoint."
        )

    # Identify the model from its parameter names rather than assuming
    # every checkpoint is MobileNetV3-Small.
    keys = set(state_dict.keys())
    if any(key.startswith(("stage2.", "stage3.", "stage4.")) for key in keys)             or any(key.startswith("conv1.") for key in keys):
        architecture = "ShuffleNetV2"
        model = models.shufflenet_v2_x1_0(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    elif any(key.startswith("features.") for key in keys)             and any(key.startswith("classifier.") for key in keys):
        architecture = "MobileNetV3-Small"
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(
            model.classifier[3].in_features,
            len(CLASS_NAMES),
        )
    else:
        sample_keys = list(state_dict.keys())[:8]
        raise ValueError(
            "Could not identify the checkpoint architecture. Expected a "
            "MobileNetV3-Small or ShuffleNetV2 state dictionary. "
            f"Example checkpoint keys: {sample_keys}"
        )

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"The checkpoint was identified as {architecture}, but its weights "
            "do not match the expected three-class model structure. Verify that "
            "you are using the correct trained checkpoint and classifier head. "
            f"Original loading error: {exc}"
        ) from exc

    model.to(DEVICE)
    model.eval()
    return model


IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def predict_eye_image(model, image):
    """Return the predicted class, its model score, and all class scores."""
    image = image.convert("RGB")
    tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    scores = {
        CLASS_NAMES[i]: float(probabilities[i].item() * 100)
        for i in range(len(CLASS_NAMES))
    }

    predicted_index = int(torch.argmax(probabilities).item())
    prediction = CLASS_NAMES[predicted_index]
    confidence = scores[prediction]

    return prediction, confidence, scores


# =========================================================
# 5. GROQ VOICE FUNCTIONS
# =========================================================

def transcribe_voice(audio_bytes, language):
    if not groq_client:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "voice_input.wav"

    response = groq_client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-large-v3",
        language=LANGUAGES[language]["stt"],
        response_format="text",
    )

    if isinstance(response, str):
        return response.strip()

    return response.text.strip()


def answer_voice_question(
    question,
    prediction,
    confidence,
    language,
):
    if not groq_client:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    completion = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are a safe eye-screening assistant. "
                    f"Respond in {language}, using its native script. "
                    f"The AI screening model predicted {prediction}. "
                    f"Its displayed score is {confidence:.1f} percent. "
                    "Explain this result in simple, accessible language. "
                    "This is an experimental screening result, not a "
                    "confirmed diagnosis. Do not claim certainty, prescribe "
                    "medicines, or recommend treatment without a clinician. "
                    "Recommend evaluation by a qualified eye-care professional. "
                    "Sudden vision loss, severe eye pain, or serious eye injury "
                    "requires urgent medical attention. If the user asks an "
                    "unrelated question, politely explain your eye-health focus."
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        temperature=0.2,
        max_tokens=350,
    )

    return completion.choices[0].message.content.strip()


def create_voice_response(text, language):
    audio_buffer = io.BytesIO()

    speech = gTTS(
        text=text,
        lang=LANGUAGES[language]["tts"],
        slow=False,
    )

    speech.write_to_fp(audio_buffer)
    audio_buffer.seek(0)

    return audio_buffer.read()


# =========================================================
# 6. PDF REPORT
# =========================================================

def create_screening_pdf(
    prediction,
    confidence,
    explanation,
    language,
):
    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45,
    )

    styles = getSampleStyleSheet()
    story = []

    story.append(
        Paragraph(
            "Rural Eye Screening AI Assistant",
            styles["Title"],
        )
    )
    story.append(Spacer(1, 0.2 * inch))

    story.append(
        Paragraph(
            f"Date and time: {datetime.now().strftime('%d-%m-%Y %H:%M')}",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 0.15 * inch))

    display_prediction = LANGUAGES[language]["diseases"].get(
        prediction,
        prediction.title(),
    )

    story.append(
        Paragraph(
            f"Screening result: {display_prediction}",
            styles["Heading2"],
        )
    )

    story.append(
        Paragraph(
            f"Model score: {confidence:.2f}%",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 0.2 * inch))

    story.append(
        Paragraph(
            "AI explanation",
            styles["Heading2"],
        )
    )

    # ReportLab's default fonts do not support all scripts.
    # Avoid producing unreadable boxes for non-Latin text.
    if explanation and explanation.isascii():
        safe_explanation = (
            explanation.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br/>")
        )
        story.append(
            Paragraph(safe_explanation, styles["BodyText"])
        )
    elif explanation:
        story.append(
            Paragraph(
                "The explanation is displayed in the selected language "
                "in the application. This PDF's default font cannot "
                "reliably render that script.",
                styles["BodyText"],
            )
        )
    else:
        story.append(
            Paragraph(
                "No AI explanation was generated.",
                styles["BodyText"],
            )
        )

    story.append(Spacer(1, 0.25 * inch))
    story.append(
        Paragraph(
            "Medical notice",
            styles["Heading2"],
        )
    )
    story.append(
        Paragraph(
            "This report is generated by an experimental AI screening "
            "prototype. It is not a confirmed diagnosis and does not "
            "replace professional eye examination. Consult a qualified "
            "eye-care professional. Seek urgent medical care for sudden "
            "vision loss, severe eye pain, or serious eye injury.",
            styles["BodyText"],
        )
    )

    document.build(story)
    buffer.seek(0)

    return buffer.getvalue()


# =========================================================
# 7. SESSION STATE
# =========================================================

DEFAULTS = {
    "site_language": "English",
    "screening_result": None,
    "screening_explanation": None,
    "explanation_language": None,
    "uploaded_image_hash": None,
    "voice_question": None,
    "voice_answer": None,
    "voice_audio": None,
    "voice_audio_hash": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# 8. LANGUAGE SELECTION AND PAGE HEADER
# =========================================================

selected_language = st.selectbox(
    "🌐 Language / ಭಾಷೆ / भाषा",
    options=list(LANGUAGES.keys()),
    index=list(LANGUAGES.keys()).index(
        st.session_state.site_language
    ),
    key="language_selector",
)

st.session_state.site_language = selected_language
T = UI_TEXT[selected_language]

# Apply right-to-left alignment for Urdu.
if selected_language == "Urdu (اردو)":
    st.markdown(
        """
        <style>
        .main .block-container {
            direction: rtl;
            text-align: right;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

st.title("👁️ " + T["title"])
st.write(T["subtitle"])

with st.expander(T["about"], expanded=False):
    st.write(T["about_text"])


# =========================================================
# 9. IMAGE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    T["upload"],
    type=["jpg", "jpeg", "png"],
    help=T["upload_help"],
)

if uploaded_file is None:
    st.info(T["upload_first"])

else:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        image_bytes = uploaded_file.getvalue()
        current_hash = hashlib.sha256(image_bytes).hexdigest()

        # Clear old result and voice responses when the image changes.
        if current_hash != st.session_state.uploaded_image_hash:
            st.session_state.screening_result = None
            st.session_state.screening_explanation = None
            st.session_state.explanation_language = None
            st.session_state.voice_question = None
            st.session_state.voice_answer = None
            st.session_state.voice_audio = None
            st.session_state.voice_audio_hash = None

        st.session_state.uploaded_image_hash = current_hash

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(
                image,
                caption=T["upload"],
                use_container_width=True,
            )

        with col2:
            st.write("**" + T["upload"] + "**")
            st.write(
                f"{image.width} × {image.height} pixels"
            )

        if st.button(T["screen"], type="primary"):
            try:
                with st.spinner(T["processing"]):
                    model = load_eye_model(MODEL_PATH)
                    prediction, confidence, scores = predict_eye_image(
                        model,
                        image,
                    )

                st.session_state.screening_result = {
                    "prediction": prediction,
                    "confidence": confidence,
                    "scores": scores,
                }

                st.session_state.screening_explanation = None
                st.session_state.explanation_language = None
                st.session_state.voice_question = None
                st.session_state.voice_answer = None
                st.session_state.voice_audio = None
                st.session_state.voice_audio_hash = None

            except Exception as e:
                st.error(f"{T['error']}: {e}")

    except Exception as e:
        st.error(f"{T['invalid']} {e}")


# =========================================================
# 10. SCREENING RESULT
# =========================================================

result = st.session_state.screening_result

if result:
    prediction = result["prediction"]
    confidence = result["confidence"]

    display_prediction = LANGUAGES[selected_language]["diseases"].get(
        prediction,
        prediction.title(),
    )

    st.divider()
    st.header(T["result"])

    result_col, score_col = st.columns(2)

    with result_col:
        st.metric(
            label=T["result"],
            value=display_prediction,
        )

    with score_col:
        st.metric(
            label=T["confidence"],
            value=f"{confidence:.2f}%",
        )

    st.caption(T["confidence_note"])

    # Show every class score to help inspect confusing predictions.
    # These are model scores, not calibrated clinical probabilities.
    class_scores = result.get("scores", {})
    if class_scores:
        st.subheader("Class score breakdown")
        for class_name in CLASS_NAMES:
            score = class_scores.get(class_name)
            if score is not None:
                st.write(f"**{class_name.title()}** — {score:.2f}%")
                st.progress(max(0, min(100, int(round(score)))))

        sorted_scores = sorted(
            class_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if len(sorted_scores) >= 2:
            score_gap = sorted_scores[0][1] - sorted_scores[1][1]
            if score_gap < 10:
                st.warning(
                    "The two highest class scores are close. "
                    "Treat this result as uncertain and seek review by "
                    "a qualified eye-care professional."
                )

    if prediction == "normal":
        st.info(
            "The model classified this image as normal. "
            "This does not rule out an eye condition."
        )
    else:
        st.warning(
            "The model detected features associated with "
            f"{display_prediction}. This is not a confirmed diagnosis; "
            "please arrange an eye examination."
        )

    st.divider()
    st.subheader(T["explanation"])

    if st.session_state.explanation_language != selected_language:
        if st.session_state.screening_explanation:
            st.info(T["language_changed"])

        if st.button(T["generate"]):
            if not groq_client:
                st.error(T["api_missing"])
            else:
                try:
                    with st.spinner(T["generating"]):
                        explanation = explain_screening_result(
                            prediction=prediction,
                            confidence=confidence,
                            language=selected_language,
                        )

                    st.session_state.screening_explanation = explanation
                    st.session_state.explanation_language = selected_language
                    st.rerun()

                except Exception as e:
                    st.error(f"{T['error']}: {e}")

    elif st.session_state.screening_explanation:
        st.write(st.session_state.screening_explanation)

    else:
        if st.button(T["generate"]):
            if not groq_client:
                st.error(T["api_missing"])
            else:
                try:
                    with st.spinner(T["generating"]):
                        explanation = explain_screening_result(
                            prediction=prediction,
                            confidence=confidence,
                            language=selected_language,
                        )

                    st.session_state.screening_explanation = explanation
                    st.session_state.explanation_language = selected_language
                    st.rerun()

                except Exception as e:
                    st.error(f"{T['error']}: {e}")


    # =====================================================
    # 11. VOICE ASSISTANT
    # =====================================================

    st.divider()
    st.header("🎙️ " + T["voice_title"])
    st.write(T["voice_help"])

    if not groq_client:
        st.warning(T["api_missing"])

    else:
        audio_bytes = audio_recorder(
            text=T["record"],
            recording_color="#e74c3c",
            neutral_color="#2563eb",
            icon_name="microphone",
            icon_size="2x",
            key="voice_recorder",
        )

        if audio_bytes:
            audio_hash = hashlib.sha256(audio_bytes).hexdigest()

            # Process each recording only once.
            if audio_hash != st.session_state.voice_audio_hash:
                try:
                    with st.spinner(T["transcribing"]):
                        question = transcribe_voice(
                            audio_bytes,
                            selected_language,
                        )

                    if not question:
                        st.warning(
                            "No speech was recognized. Please record again."
                        )
                    else:
                        st.write(
                            f"**{T['you_asked']}:** {question}"
                        )

                        with st.spinner(T["answering"]):
                            answer = answer_voice_question(
                                question=question,
                                prediction=prediction,
                                confidence=confidence,
                                language=selected_language,
                            )

                            spoken_audio = create_voice_response(
                                answer,
                                selected_language,
                            )

                        st.session_state.voice_question = question
                        st.session_state.voice_answer = answer
                        st.session_state.voice_audio = spoken_audio
                        st.session_state.voice_audio_hash = audio_hash

                except Exception as e:
                    st.error(f"{T['error']}: {e}")

        if st.session_state.voice_answer:
            st.subheader(T["voice_answer"])
            st.write(st.session_state.voice_answer)

            if st.session_state.voice_audio:
                st.audio(
                    st.session_state.voice_audio,
                    format="audio/mp3",
                )


    # =====================================================
    # 12. DOWNLOADABLE PDF REPORT
    # =====================================================

    st.divider()
    st.subheader(T["pdf_title"])

    explanation = (
        st.session_state.screening_explanation
        if st.session_state.explanation_language == selected_language
        else None
    )

    try:
        pdf_bytes = create_screening_pdf(
            prediction=prediction,
            confidence=confidence,
            explanation=explanation,
            language=selected_language,
        )

        st.download_button(
            label=T["pdf_button"],
            data=pdf_bytes,
            file_name="rural_eye_screening_report.pdf",
            mime="application/pdf",
        )

    except Exception as e:
        st.error(f"{T['error']}: {e}")


# =========================================================
# 13. MEDICAL NOTICE AND FOOTER
# =========================================================

st.divider()
st.subheader("⚠️ " + T["disclaimer"])
st.warning(T["disclaimer_text"])

st.caption(T["footer"])
