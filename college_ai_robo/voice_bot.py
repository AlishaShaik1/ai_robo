import speech_recognition as sr
import pyttsx3
import logging
import time
import platform
import subprocess

# Import the response logic from your existing app
try:
    from app import respond
    print("Successfully imported logic from app.py")
except ImportError as e:
    print(f"Error importing app.py: {e}")
    print("Make sure app.py is in the same directory.")
    exit(1)

# ==========================
# INITIALIZATION
# ==========================
recognizer = sr.Recognizer()
recognizer.energy_threshold = 300
recognizer.dynamic_energy_threshold = True

def clean_text_for_speech(text):
    """Remove markdown and special characters for better speech."""
    text = text.replace("**", "").replace("__", "").replace("*", "")
    text = text.replace("`", "")
    return text

# ==========================
# ✅ UPDATED VOICE FUNCTION (ONLY CHANGE)
# ==========================
def speak(text):
    """Convert text to speech (Female voice on Raspberry Pi & Windows)."""
    try:
        clean_text = clean_text_for_speech(text)
        print(f"Robot: {text}")

        # Raspberry Pi / Linux → MBROLA female voice
        if platform.system() == "Linux":
            subprocess.run(
                ["espeak-ng", "-v", "mb-us1", "-s", "165", clean_text],
                check=False
            )

        # Windows / Laptop → pyttsx3 female (Zira)
        else:
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            for voice in voices:
                if "female" in voice.name.lower() or "zira" in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 1.0)
            engine.say(clean_text)
            engine.runAndWait()
            engine.stop()

    except Exception as e:
        print(f"TTS Error: {e}")

# ==========================
# LISTEN FUNCTION
# ==========================
def listen_for_command(source):
    recognizer.pause_threshold = 1.5
    recognizer.dynamic_energy_adjustment_ratio = 1.5

    print("Listening... (Say 'chitti' to wake me up)")
    try:
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=20)
        print("Recognizing...")
        command = recognizer.recognize_google(audio, language='en-IN')
        print(f"User said: {command}")
        return command.lower()

    except sr.WaitTimeoutError:
        return None
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        speak("I am having trouble connecting to the internet.")
        return None
    except Exception as e:
        print(f"Microphone Error: {e}")
        return None

# ==========================
# OPENAI CLOUD MODE
# ==========================
import openai
openai.api_key = "YOUR_API_KEY_HERE"

def ask_openai(prompt):
    try:
        print(f"Connecting to OpenAI: {prompt}")
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are Chitti, a helpful AI assistant. Keep answers concise."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI Error: {e}")
        return "I am having trouble connecting to the cloud."

# ==========================
# MAIN LOOP
# ==========================
def main():
    speak("Hello, I am ready. Say Chitti for college info, or Hey Chitti for general questions.")

    LOCAL_WAKE_WORDS = ["chitti", "city", "chiti", "chithi", "chetty", "chilly", "giti", "shitti"]
    OPENAI_WAKE_WORDS = ["hey chitti", "hey city", "hi chitti", "hi city", "hey chetty"]

    try:
        with sr.Microphone() as source:
            print("\nAdjusting for ambient noise... (Please wait)")
            recognizer.adjust_for_ambient_noise(source, duration=2)
            print("Ready. Listening...")

            listening = True
            while listening:
                command = listen_for_command(source)

                if command:
                    target_mode = None
                    query = ""

                    for alias in OPENAI_WAKE_WORDS:
                        if alias in command:
                            target_mode = "cloud"
                            query = command.replace(alias, "", 1).strip()
                            break

                    if not target_mode:
                        for alias in LOCAL_WAKE_WORDS:
                            if alias in command:
                                target_mode = "local"
                                query = command.replace(alias, "", 1).strip()
                                break

                    if target_mode:
                        if not query:
                            speak("Yes?")
                            continue

                        if "exit" in query or "quit" in query or "stop" in query:
                            speak("Goodbye!")
                            listening = False
                            break

                        if target_mode == "cloud":
                            print(f"Mode: CLOUD | Query: {query}")
                            response = ask_openai(query)
                            speak(response)
                        else:
                            print(f"Mode: LOCAL | Query: {query}")
                            try:
                                response_text = respond(query, [])
                                speak(response_text)
                            except Exception as e:
                                print(f"Processing Error: {e}")
                                speak("Local processing error.")
                    else:
                        print(f"Ignored: {command}")

    except KeyboardInterrupt:
        print("\nStopping...")

if __name__ == "__main__":
    main()
