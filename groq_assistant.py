
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

LANGUAGES = {
    "English": "English, using the Latin script",
    "Hindi (हिंदी)": "Hindi, using the Devanagari script",
    "Punjabi (ਪੰਜਾਬੀ)": "Punjabi, using the Gurmukhi script",
    "Urdu (اردو)": "Urdu, using the Urdu Perso-Arabic script, written right to left",
    "Kannada (ಕನ್ನಡ)": "Kannada, using the Kannada script",
    "Tamil (தமிழ்)": "Tamil, using the Tamil script",
    "Telugu (తెలుగు)": "Telugu, using the Telugu script",
}


def explain_screening_result(prediction, confidence, language="English"):
    """Generate a patient-friendly explanation in the selected language."""

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return (
            "AI explanation is unavailable because the Groq API key "
            "was not found. Please check your .env configuration."
        )

    language_instruction = LANGUAGES.get(
        language,
        LANGUAGES["English"],
    )

    try:
        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a multilingual patient-education "
                        "assistant for an experimental eye-screening "
                        "research application.\n\n"
                        f"RESPONSE LANGUAGE: {language_instruction}.\n"
                        "Write the entire patient-facing explanation "
                        "in the selected language and its native script. "
                        "Do not transliterate into English letters. "
                        "Do not switch to English except for unavoidable "
                        "technical terms, and explain those terms simply.\n\n"
                        "Explain the screening result in simple language "
                        "that a person without medical training can "
                        "understand. Explain what the predicted class "
                        "generally means and recommend evaluation by a "
                        "qualified ophthalmologist.\n\n"
                        "Never claim the prediction is a confirmed "
                        "diagnosis. Never prescribe medicines or "
                        "treatment. Explain that the model score is "
                        "not a calibrated medical probability.\n\n"
                        "Mention that sudden vision loss, severe eye "
                        "pain, or serious eye injury requires urgent "
                        "medical assessment. Keep the explanation "
                        "concise, respectful, and culturally neutral."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Predicted class: "
                        f"{prediction.replace('_', ' ').title()}\n"
                        f"Model score: {confidence * 100:.2f}%\n\n"
                        "Explain this experimental screening result "
                        "in the selected language. Include a simple "
                        "explanation and the appropriate next step."
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=400,
        )

        explanation = response.choices[0].message.content

        if explanation:
            return explanation

        return (
            "The AI service did not return an explanation. "
            "Please try again."
        )

    except Exception as error:
        print("Groq explanation error:", error)

        return (
            "The AI explanation could not be generated. "
            "Please try again later or consult an ophthalmologist."
        )
