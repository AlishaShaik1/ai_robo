import speech_recognition as sr
import pyttsx3
import logging
import time

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
# Initialize Recognizer
recognizer = sr.Recognizer()
# Set energy threshold explicitly (good default), dynamic will adjust from here
recognizer.energy_threshold = 300
recognizer.dynamic_energy_threshold = True

def clean_text_for_speech(text):
    """Remove markdown and special characters for better speech."""
    # Remove bold/italic markers
    text = text.replace("**", "").replace("__", "").replace("*", "")
    # Remove code blocks if any (simple approach)
    text = text.replace("`", "")
    return text

def speak(text):
    """Convert text to speech."""
    try:
        clean_text = clean_text_for_speech(text)
        print(f"Robot: {text}") # Print original with formatting
        
        # Re-initialize engine each time to avoid 'runAndWait' hangs in loops
        engine = pyttsx3.init()
        
        # Configure Voice
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
        del engine # Clean up
    except Exception as e:
        print(f"TTS Error: {e}")

def listen_for_command(source):
    """Listen to the microphone and return recognized text. Uses the passed source."""
    # Note: adjust_for_ambient_noise is now done ONCE in main()
    
    # Increase pause threshold to allow for gaps in speech (User requested specifically for large input)
    recognizer.pause_threshold = 1.5 
    # Dynamic energy ratio
    recognizer.dynamic_energy_adjustment_ratio = 1.5
    
    print("Listening... (Say 'chitti' to wake me up)")
    
    try:
        # Listen with a timeout to prevent hanging forever if no one speaks
        # phrase_time_limit=20 allows for long rank queries
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=20)
        
        print("Recognizing...")
        # Use simple English first, or 'en-IN' for Indian accents
        command = recognizer.recognize_google(audio, language='en-IN')
        print(f"User said: {command}")
        return command.lower()
        
    except sr.WaitTimeoutError:
        return None
    except sr.UnknownValueError:
        # print("Could not understand audio.") # Reduced log spam
        return None
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        speak("I am having trouble connecting to the internet.")
        return None
    except Exception as e:
        print(f"Microphone Error: {e}")
        return None

import openai

# Configure OpenAI API Key
# Replace 'YOUR_API_KEY' with your actual key or set os.environ["OPENAI_API_KEY"]
openai.api_key = "YOUR_API_KEY_HERE"

def ask_openai(prompt):
    """Fetch response from OpenAI GPT."""
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

def main():
    speak("Hello, I am ready. Say 'Chitti' for college info, or 'Hey Chitti' for general questions.")
    
    # Wake Word Aliases
    LOCAL_WAKE_WORDS = ["chitti", "city", "chiti", "chithi", "chetty", "chilly", "giti", "shitti"]
    OPENAI_WAKE_WORDS = ["hey chitti", "hey city", "hi chitti", "hi city", "hey chetty"]
    
    # Keep microphone open
    try:
        with sr.Microphone() as source:
            print("\nAdjusting for ambient noise... (Please wait)")
            recognizer.adjust_for_ambient_noise(source, duration=2)
            print("Ready. Listening...")
            
            listening = True
            while listening:
                command = listen_for_command(source)
                
                if command:
                    target_mode = None # 'local' or 'cloud'
                    query = ""
                    
                    # 1. Check Cloud Wake Word ("Hey Chitti") first (Longer match priority)
                    for alias in OPENAI_WAKE_WORDS:
                        if alias in command:
                            target_mode = 'cloud'
                            query = command.replace(alias, "", 1).strip()
                            break
                            
                    # 2. If not cloud, check Local Wake Word ("Chitti")
                    if not target_mode:
                        for alias in LOCAL_WAKE_WORDS:
                            if alias in command:
                                target_mode = 'local'
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
                        
                        if target_mode == 'cloud':
                            print(f"Mode: CLOUD (OpenAI) | Query: {query}")
                            gpt_response = ask_openai(query)
                            speak(gpt_response)
                        else:
                            print(f"Mode: LOCAL (College DB) | Query: {query}")
                            try:
                                response_text = respond(query, [])
                                speak(response_text)
                            except Exception as e:
                                print(f"Processing Error: {e}")
                                speak("Local processing error.")
                    
                    else:
                        print(f"Ignored: '{command}' (No wake word)")
                        
    except KeyboardInterrupt:
        print("\nStopping...")

if __name__ == "__main__":
    main()