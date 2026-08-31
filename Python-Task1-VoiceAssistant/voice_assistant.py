import speech_recognition as sr
import pyttsx3
import datetime
import webbrowser

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty("rate", 170)


def speak(text):
    print("Assistant:", text)
    engine.say(text)
    engine.runAndWait()


def listen():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("\nListening...")
        recognizer.adjust_for_ambient_noise(source, duration=1)

        try:
            audio = recognizer.listen(
                source,
                timeout=5,
                phrase_time_limit=8
            )
        except sr.WaitTimeoutError:
            speak("I did not hear anything. Please try again.")
            return ""

    try:
        print("Recognizing...")
        command = recognizer.recognize_google(audio)
        print("You:", command)
        return command.lower()

    except sr.UnknownValueError:
        speak("Sorry, I could not understand you. Please repeat.")
        return ""

    except sr.RequestError:
        speak("Sorry, the speech recognition service is unavailable.")
        return ""


def handle_command(command):

    if "hello" in command or "hi" in command:
        speak("Hello! How can I help you?")

    elif "time" in command:
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        speak(f"The current time is {current_time}.")

    elif "date" in command:
        current_date = datetime.datetime.now().strftime("%B %d, %Y")
        speak(f"Today's date is {current_date}.")

    elif "search" in command:
        search_query = command.replace("search", "").strip()

        if search_query:
            speak(f"Searching the web for {search_query}.")
            url = (
                "https://www.google.com/search?q="
                + search_query.replace(" ", "+")
            )
            webbrowser.open(url)
        else:
            speak("What would you like me to search for?")

    elif "exit" in command or "quit" in command or "stop" in command:
        speak("Goodbye!")
        return False

    elif command:
        speak(
            "I don't know that command yet. "
            "Please try hello, time, date, or search."
        )

    return True


def main():
    speak("Voice assistant started.")
    speak("You can say hello, ask for the time or date, "
          "or search the web.")
    speak("Say exit to close the assistant.")

    running = True

    while running:
        command = listen()

        if command:
            running = handle_command(command)


if __name__ == "__main__":
    main()
