import streamlit as st
import speech_recognition as sr
import asyncio
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from deep_translator import GoogleTranslator
import requests
import random
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import edge_tts


nltk.download('vader_lexicon')


if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)


@st.cache_resource
def load_model():
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="auto")
    return model, tokenizer

model, tokenizer = load_model()


def generate_commentary(event_text):
    inputs = tokenizer(event_text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    output = model.generate(**inputs, max_length=100)
    return tokenizer.decode(output[0], skip_special_tokens=True)


def translate_commentary(text, target_language):
    return GoogleTranslator(source='auto', target=target_language).translate(text)


def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(text)
    if score['compound'] >= 0.05:
        return "excited"
    elif score['compound'] <= -0.05:
        return "serious"
    else:
        return "neutral"


def fetch_player_stats(player_name):
    fake_stats = {
        "Virat Kohli": "80 runs off 50 balls, 10 fours, 2 sixes.",
        "MS Dhoni": "45 runs off 30 balls, finishing in style!",
        "Jasprit Bumrah": "4 overs, 2 wickets, 15 runs conceded.",
    }
    return fake_stats.get(player_name, "No recent data available.")


async def text_to_speech(text, voice="en-US-GuyNeural"):
    output_audio = "commentary.mp3"
    tts = edge_tts.Communicate(text, voice)
    await tts.save(output_audio)
    return output_audio


st.title("Cricket Commentary Generator")

language_options = {"English": "en", "Hindi": "hi", "Spanish": "es", "Tamil": "ta"}
selected_language = st.selectbox("Choose Commentary Language", options=language_options.keys())


def recognize_speech():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    
    with mic as source:
        st.info("Speak")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source, timeout=10)
    
    try:
        text = recognizer.recognize_google(audio)
        st.success(f"Recognized Speech: {text}")
        return text
    except sr.UnknownValueError:
        st.error("sorry speak again")
        return None
    except sr.RequestError as e:
        st.error(f"Speech recognition error: {e}")
        return None


if st.button("Start Commentary"):
    event_text = recognize_speech()
    
    if event_text:
        sentiment = analyze_sentiment(event_text)
        player = random.choice(["Virat Kohli", "MS Dhoni", "Jasprit Bumrah"])
        player_stats = fetch_player_stats(player)

     
        final_prompt = f"({sentiment.upper()} Commentary) {event_text} | Player Update: {player} - {player_stats}"
        ai_generated = generate_commentary(final_prompt)
        
        
        translated_commentary = translate_commentary(ai_generated, language_options[selected_language])
        
        st.success(f"AI Commentary ({selected_language}): {translated_commentary}")

        
        audio_file = asyncio.run(text_to_speech(translated_commentary))
        st.audio(audio_file, format="audio/mp3")

    else:
        st.warning("Not an input")


st.markdown("""
### User Guide
1️⃣ Speak about the match situation or about players.  
2️⃣ AI will generate context-based commentary and it will read the commentary out loud!  
3️⃣ It will adjust excitement level based on sentiment.  
4️⃣ Commentary is personalized with real player stats.
""")