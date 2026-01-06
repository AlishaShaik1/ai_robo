import speech_recognition as sr
import pyttsx3
import logging
import time
import platform
import subprocess
import os
import uuid
import threading
import hashlib
CACHE_DIR = "/tmp/chitti_tts_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
import signal
from gtts import gTTS

# ==========================
# GPIO SETUP (NEW FEATURE)
# ==========================
import RPi.GPIO as GPIO

RED_LED = 17    # Listening
GREEN_LED = 19  # Speaking

GPIO.setmode(GPIO.BCM)
GPIO.setup(RED_LED, GPIO.OUT)
GPIO.setup(GREEN_LED, GPIO.OUT)
GPIO.output(RED_LED, GPIO.LOW)
GPIO.output(GREEN_LED, GPIO.LOW)

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

# Global variables for stop control
stop_speaking = False
is_speaking = False
audio_process = None
festival_process = None

def force_stop_speaking():
    """Immediately stop current speech."""
    global stop_speaking, is_speaking, audio_process, festival_process

    stop_speaking = True
    is_speaking = False

    if festival_process and festival_process.poll() is None:
        try:
            festival_process.terminate()
        except Exception:
            pass

    if audio_process and audio_process.poll() is None:
        try:
            audio_process.terminate()
            audio_process.wait()
        except Exception:
            pass

    try:
        from pygame import mixer
        if mixer.get_init() and mixer.music.get_busy():
            mixer.music.stop()
            mixer.music.unload()
    except Exception:
        pass

    GPIO.output(RED_LED, GPIO.LOW)
    GPIO.output(GREEN_LED, GPIO.LOW)

    print("[✓ Speech force-stopped]")

def clean_text_for_speech(text):
    text = text.replace("**", "").replace("__", "").replace("*", "")
    text = text.replace("`", "")
    return text

# ==========================
# 🎙️ FESTIVAL TTS (ONLY CHANGE)
# ==========================
def speak(text, mic_source=None):
    """Convert text to speech using Google gTTS with interrupt support + caching."""
    global stop_speaking, is_speaking, audio_process

    try:
        stop_speaking = False  # Reset flag
        is_speaking = True  # Mark that we're speaking
        
        clean_text = clean_text_for_speech(text)
        print(f"Robot: {text}")

        # 🔥 CACHE LOGIC (KEY PART)
        text_hash = hashlib.md5(clean_text.encode()).hexdigest()
        filename = os.path.join(CACHE_DIR, f"{text_hash}.mp3")

        # Generate audio ONLY if not cached
        if not os.path.exists(filename):
            tts = gTTS(text=clean_text, lang="en", slow=False)
            tts.save(filename)

        # ======================
        # PLAY AUDIO (LINUX / PI)
        # ======================
        if platform.system() == "Linux":
            audio_process = subprocess.Popen(
                ["mpg123", "-q", filename],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            while audio_process.poll() is None:
                if stop_speaking:
                    print("[Speech interrupted]")
                    audio_process.terminate()
                    audio_process.wait()
                    break
                time.sleep(0.1)

        # ======================
        # WINDOWS PLAYBACK
        # ======================
        else:
            try:
                from pygame import mixer
                mixer.init()
                mixer.music.load(filename)
                mixer.music.play()

                while mixer.music.get_busy():
                    if stop_speaking:
                        print("[Speech interrupted]")
                        mixer.music.stop()
                        break
                    time.sleep(0.1)

                mixer.music.unload()

            except ImportError:
                engine = pyttsx3.init()
                engine.say(clean_text)
                engine.runAndWait()
                engine.stop()

        # Mark speech complete
        is_speaking = False

    except Exception as e:
        print(f"TTS Error: {e}")
        is_speaking = False

# ==========================
# LISTEN FUNCTION
# ==========================
def listen_for_command(source):
    recognizer.pause_threshold = 1.5
    recognizer.dynamic_energy_adjustment_ratio = 1.5

    GPIO.output(RED_LED, GPIO.HIGH)
    print("Listening... (Say 'chitti' to wake me up)")

    try:
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=20)
        GPIO.output(RED_LED, GPIO.LOW)
        print("Recognizing...")
        command = recognizer.recognize_google(audio, language='en-IN')
        print(f"User said: {command}")
        return command.lower()

    except sr.WaitTimeoutError:
        GPIO.output(RED_LED, GPIO.LOW)
        return None
    except sr.UnknownValueError:
        GPIO.output(RED_LED, GPIO.LOW)
        return None
    except sr.RequestError as e:
        GPIO.output(RED_LED, GPIO.LOW)
        speak("I am having trouble connecting to the internet.")
        return None
    except Exception as e:
        GPIO.output(RED_LED, GPIO.LOW)
        print(f"Microphone Error: {e}")
        return None

# ==========================
# BACKGROUND STOP LISTENER
# ==========================
def listen_for_stop(source):
    global stop_speaking, is_speaking

    stop_recognizer = sr.Recognizer()
    stop_recognizer.pause_threshold = 0.5
    stop_recognizer.energy_threshold = 400

    print("[Stop listener active]")

    while is_speaking and not stop_speaking:
        try:
            audio = stop_recognizer.listen(source, timeout=0.3, phrase_time_limit=2)
            command = stop_recognizer.recognize_google(audio, language='en-IN').lower()
            if any(word in command for word in ["stop", "chitti stop", "city stop", "quiet", "silence", "shut up"]):
                force_stop_speaking()
                break
        except Exception:
            continue

# ==========================
# OPENAI CLOUD MODE
# ==========================
import openai
openai.api_key = "YOUR_API_KEY_HERE"

def ask_openai(prompt):
    try:
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

    LOCAL_WAKE_WORDS = ["chitti", "city", "chiti", "chithi", "chetty", "chilly", "giti", "shitti", "chinti", "chinki", "shakti", "shanti", "pretty"]
    OPENAI_WAKE_WORDS = ["hey chitti", "hey city", "hi chitti", "hi city", "hey chetty", "hey chinti", "hey shanti"]

    try:
        with sr.Microphone() as source:
            print("\nAdjusting for ambient noise... (Please wait)")
            recognizer.adjust_for_ambient_noise(source, duration=2)
            print("Ready. Listening...")

            listening = True
            while listening:
                command = listen_for_command(source)

                if command and is_speaking:
                    if any(w in command for w in ["chitti stop", "city stop", "stop", "quiet", "silence", "shut up"]):
                        force_stop_speaking()
                        continue

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

                        if any(w in query for w in ["exit", "quit", "bye"]):
                            speak("Goodbye!")
                            listening = False
                            break

                        if target_mode == "cloud":
                            print(f"Mode: CLOUD | Query: {query}")
                            response = ask_openai(query)
                            threading.Thread(target=speak, args=(response, source), daemon=True).start()
                        else:
                            print(f"Mode: LOCAL | Query: {query}")
                            try:
                                response_text = respond(query, [])
                                threading.Thread(target=speak, args=(response_text, source), daemon=True).start()
                            except Exception as e:
                                print(f"Processing Error: {e}")
                                speak("Local processing error.")
                    else:
                        print(f"Ignored: {command}")

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        GPIO.output(RED_LED, GPIO.LOW)
        GPIO.output(GREEN_LED, GPIO.LOW)
        GPIO.cleanup()

if __name__ == "__main__":
    main()
