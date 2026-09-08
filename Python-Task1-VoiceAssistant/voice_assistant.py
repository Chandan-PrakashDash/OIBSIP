import speech_recognition as sr
import pyttsx3
import datetime
import webbrowser
import requests
import threading
import time
import smtplib
import os
import re
import nltk

from nltk.classify import NaiveBayesClassifier
from email.message import EmailMessage
from urllib.parse import quote


# ============================================================
# VOICE ASSISTANT - OIBSIP TASK 1
# Beginner + Advanced Features
# ============================================================


# ------------------------------------------------------------
# NLTK SETUP
# ------------------------------------------------------------

# We use wordpunct_tokenize so that no additional NLTK
# tokenizer download is required.
#
# NLTK is used for Natural Language Intent Recognition.

def extract_features(sentence):
    """
    Convert a sentence into features for the NLTK classifier.
    """

    words = nltk.wordpunct_tokenize(
        sentence.lower()
    )

    # Keep only useful words
    words = [
        word for word in words
        if word.isalnum()
    ]

    return {
        word: True
        for word in words
    }


# ------------------------------------------------------------
# TRAINING DATA FOR NATURAL LANGUAGE INTENTS
# ------------------------------------------------------------

intent_data = [

    # --------------------------------------------------------
    # GREETING
    # --------------------------------------------------------

    ("hello", "greeting"),
    ("hi", "greeting"),
    ("hey", "greeting"),
    ("hello assistant", "greeting"),
    ("hi assistant", "greeting"),
    ("hey assistant", "greeting"),
    ("good morning", "greeting"),
    ("good afternoon", "greeting"),
    ("good evening", "greeting"),
    ("nice to meet you", "greeting"),
    ("how are you", "greeting"),
    ("hello how are you", "greeting"),


    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    ("what time is it", "time"),
    ("tell me the time", "time"),
    ("what is the current time", "time"),
    ("tell me the current time", "time"),
    ("can you tell me the time", "time"),
    ("could you tell me the time", "time"),
    ("please tell me the time", "time"),
    ("do you know the time", "time"),
    ("would you tell me what time it is", "time"),
    ("could you please tell me what time it is", "time"),
    ("what time is it right now", "time"),
    ("what is the time right now", "time"),


    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    ("what is today's date", "date"),
    ("what is todays date", "date"),
    ("tell me today's date", "date"),
    ("tell me todays date", "date"),
    ("what date is it today", "date"),
    ("what is the current date", "date"),
    ("tell me the current date", "date"),
    ("can you tell me today's date", "date"),
    ("could you tell me the date", "date"),
    ("what day is it today", "date"),


    # --------------------------------------------------------
    # WEATHER
    # --------------------------------------------------------

    ("what is the weather", "weather"),
    ("tell me the weather", "weather"),
    ("what is the weather today", "weather"),
    ("how is the weather today", "weather"),
    ("what is the weather like", "weather"),
    ("can you tell me the weather", "weather"),
    ("could you tell me the weather", "weather"),
    ("what is the temperature", "weather"),
    ("tell me the temperature", "weather"),
    ("how hot is it", "weather"),
    ("how cold is it", "weather"),
    ("what is the forecast", "weather"),
    ("tell me the forecast", "weather"),
    ("what is the weather like today", "weather"),
    ("can you tell me how the weather is", "weather"),


    # --------------------------------------------------------
    # REMINDER
    # --------------------------------------------------------

    ("set a reminder", "reminder"),
    ("please set a reminder", "reminder"),
    ("can you set a reminder", "reminder"),
    ("could you set a reminder", "reminder"),
    ("I need a reminder", "reminder"),
    ("remind me later", "reminder"),
    ("remind me in ten minutes", "reminder"),
    ("remind me in five minutes", "reminder"),
    ("remind me after ten minutes", "reminder"),
    ("set a reminder for me", "reminder"),
    ("please remind me", "reminder"),


    # --------------------------------------------------------
    # EMAIL
    # --------------------------------------------------------

    ("send an email", "email"),
    ("send email", "email"),
    ("please send an email", "email"),
    ("can you send an email", "email"),
    ("could you send an email", "email"),
    ("I want to send an email", "email"),
    ("I need to send an email", "email"),
    ("send an email for me", "email"),
    ("send an email to someone", "email"),
    ("email someone", "email"),


    # --------------------------------------------------------
    # MUSIC
    # --------------------------------------------------------

    ("play music", "music"),
    ("play a song", "music"),
    ("please play music", "music"),
    ("please play a song", "music"),
    ("can you play music", "music"),
    ("could you play a song", "music"),
    ("I want to play music", "music"),
    ("I want to listen to music", "music"),
    ("play some music", "music"),
    ("play a new song", "music"),
    ("play the latest song", "music"),
    ("play new Odia songs", "music"),
    ("play Hindi songs", "music"),
    ("play English songs", "music"),
    ("play Arijit Singh songs", "music"),


    # --------------------------------------------------------
    # SINGER SEARCH
    # --------------------------------------------------------

    ("search singer", "singer"),
    ("search for a singer", "singer"),
    ("find a singer", "singer"),
    ("find singer", "singer"),
    ("search singer Arijit Singh", "singer"),
    ("find songs by a singer", "singer"),
    ("search songs by a singer", "singer"),
    ("songs by Arijit Singh", "singer"),
    ("find songs by Arijit Singh", "singer"),
    ("search for songs by Arijit Singh", "singer"),


    # --------------------------------------------------------
    # MOVIE SEARCH
    # --------------------------------------------------------

    ("search movie", "movie"),
    ("search movies", "movie"),
    ("search for a movie", "movie"),
    ("find a movie", "movie"),
    ("find movie", "movie"),
    ("find movies", "movie"),
    ("search movie Avatar", "movie"),
    ("find movie Avatar", "movie"),
    ("search for the movie Avatar", "movie"),


    # --------------------------------------------------------
    # WEB SEARCH
    # --------------------------------------------------------

    ("search the web", "search"),
    ("search the internet", "search"),
    ("search online", "search"),
    ("look this up online", "search"),
    ("find this online", "search"),
    ("can you search for this", "search"),
    ("please search for this", "search"),
    ("search for Python", "search"),
    ("search Python Django", "search"),
    ("google this", "search"),
    ("search Google", "search"),


    # --------------------------------------------------------
    # KNOWLEDGE
    # --------------------------------------------------------

    ("who is Albert Einstein", "knowledge"),
    ("who is Mahatma Gandhi", "knowledge"),
    ("who was Mahatma Gandhi", "knowledge"),
    ("what is Python", "knowledge"),
    ("what is artificial intelligence", "knowledge"),
    ("what is machine learning", "knowledge"),
    ("what are computers", "knowledge"),
    ("tell me about computers", "knowledge"),
    ("tell me about Python", "knowledge"),
    ("explain artificial intelligence", "knowledge"),
    ("where is India", "knowledge"),
    ("when was India independent", "knowledge"),
    ("who is the president", "knowledge"),
]


# ------------------------------------------------------------
# PREPARE TRAINING DATA
# ------------------------------------------------------------

training_data = [
    (
        extract_features(sentence),
        intent
    )
    for sentence, intent in intent_data
]


# ------------------------------------------------------------
# TRAIN NLTK CLASSIFIER
# ------------------------------------------------------------

intent_classifier = NaiveBayesClassifier.train(
    training_data
)


def detect_intent(command):
    """
    Detect the user's intent from a natural-language sentence.
    """

    if not command:
        return "unknown"

    features = extract_features(command)

    intent = intent_classifier.classify(
        features
    )

    return intent


# ============================================================
# CONFIGURATION
# ============================================================

# OpenWeatherMap API key
# Get one from:
# https://openweathermap.org/

WEATHER_API_KEY = os.getenv(
    "OPENWEATHER_API_KEY",
    ""
)


# Email configuration
# Recommended:
# Use environment variables instead of writing passwords
# directly in this file.

EMAIL_ADDRESS = os.getenv(
    "ASSISTANT_EMAIL",
    ""
)

EMAIL_PASSWORD = os.getenv(
    "ASSISTANT_EMAIL_PASSWORD",
    ""
)


# Default email receiver for testing
TEST_RECEIVER_EMAIL = os.getenv(
    "TEST_RECEIVER_EMAIL",
    ""
)


# ============================================================
# TEXT TO SPEECH
# ============================================================

engine = pyttsx3.init()

engine.setProperty(
    "rate",
    170
)


def speak(text):
    """
    Speak and print assistant response.
    """

    print(
        "Assistant:",
        text
    )

    try:

        engine.say(text)

        engine.runAndWait()

    except Exception as error:

        print(
            "Text-to-speech error:",
            error
        )


# ============================================================
# SPEECH RECOGNITION
# ============================================================

def listen():
    """
    Listen to microphone and convert speech to text.
    """

    recognizer = sr.Recognizer()

    try:

        with sr.Microphone() as source:

            print(
                "\nListening..."
            )

            # Reduce microphone background noise
            recognizer.adjust_for_ambient_noise(
                source,
                duration=1
            )

            try:

                audio = recognizer.listen(
                    source,
                    timeout=5,
                    phrase_time_limit=8
                )

            except sr.WaitTimeoutError:

                speak(
                    "I did not hear anything. "
                    "Please try again."
                )

                return ""

    except Exception as error:

        print(
            "Microphone error:",
            error
        )

        speak(
            "I could not access the microphone. "
            "Please check your microphone."
        )

        return ""

    try:

        print(
            "Recognizing..."
        )

        command = recognizer.recognize_google(
            audio
        )

        print(
            "You:",
            command
        )

        return command.lower().strip()

    except sr.UnknownValueError:

        speak(
            "Sorry, I could not understand you. "
            "Please repeat."
        )

        return ""

    except sr.RequestError:

        speak(
            "Speech recognition service is unavailable."
        )

        return ""


# ============================================================
# WEB SEARCH
# ============================================================

def web_search(query):

    if not query:

        speak(
            "What would you like me to search for?"
        )

        return

    speak(
        f"Searching the web for {query}."
    )

    url = (
        "https://www.google.com/search?q="
        + quote(query)
    )

    webbrowser.open(
        url
    )


# ============================================================
# YOUTUBE / MUSIC
# ============================================================

def play_song(command):

    command = command.lower()

    song_name = command

    # Remove common words
    phrases = [
        "play",
        "please play",
        "can you play",
        "could you play",
        "i want to play"
    ]

    for phrase in phrases:

        if song_name.startswith(phrase):

            song_name = song_name[
                len(phrase):
            ].strip()

            break

    if not song_name:

        speak(
            "Which song would you like me to play?"
        )

        return

    current_year = datetime.datetime.now().year

    # New / latest music
    if "new odia song" in song_name:

        search_query = (
            f"new Odia songs {current_year}"
        )

    elif "new odia songs" in song_name:

        search_query = (
            f"new Odia songs {current_year}"
        )

    elif "latest odia song" in song_name:

        search_query = (
            f"latest Odia songs {current_year}"
        )

    elif "latest odia songs" in song_name:

        search_query = (
            f"latest Odia songs {current_year}"
        )

    elif "new hindi song" in song_name:

        search_query = (
            f"new Hindi songs {current_year}"
        )

    elif "new hindi songs" in song_name:

        search_query = (
            f"new Hindi songs {current_year}"
        )

    elif "new english song" in song_name:

        search_query = (
            f"new English songs {current_year}"
        )

    elif "new english songs" in song_name:

        search_query = (
            f"new English songs {current_year}"
        )

    elif "new tamil songs" in song_name:

        search_query = (
            f"new Tamil songs {current_year}"
        )

    elif "new telugu songs" in song_name:

        search_query = (
            f"new Telugu songs {current_year}"
        )

    elif "new punjabi songs" in song_name:

        search_query = (
            f"new Punjabi songs {current_year}"
        )

    elif "new bengali songs" in song_name:

        search_query = (
            f"new Bengali songs {current_year}"
        )

    elif "new marathi songs" in song_name:

        search_query = (
            f"new Marathi songs {current_year}"
        )

    elif "new malayalam songs" in song_name:

        search_query = (
            f"new Malayalam songs {current_year}"
        )

    elif "latest hindi songs" in song_name:

        search_query = (
            f"latest Hindi songs {current_year}"
        )

    elif "latest english songs" in song_name:

        search_query = (
            f"latest English songs {current_year}"
        )

    elif "latest tamil songs" in song_name:

        search_query = (
            f"latest Tamil songs {current_year}"
        )

    elif "latest telugu songs" in song_name:

        search_query = (
            f"latest Telugu songs {current_year}"
        )

    elif "odia song" in song_name:

        search_query = "Odia songs"

    elif "odia songs" in song_name:

        search_query = "Odia songs"

    else:

        # Any singer, song, movie, language, genre, etc.
        search_query = song_name

    speak(
        f"Searching YouTube for {search_query}."
    )

    url = (
        "https://www.youtube.com/results?search_query="
        + quote(search_query)
    )

    webbrowser.open(
        url
    )


# ============================================================
# MOVIE SEARCH
# ============================================================

def search_movie(command):

    movie_name = command

    phrases = [
        "search movie",
        "search movies",
        "find movie",
        "find movies",
        "movie",
        "movies"
    ]

    for phrase in phrases:

        movie_name = movie_name.replace(
            phrase,
            "",
            1
        ).strip()

    if not movie_name:

        speak(
            "Which movie would you like to search for?"
        )

        return

    search_query = (
        movie_name + " movie"
    )

    speak(
        f"Searching for {search_query}."
    )

    url = (
        "https://www.google.com/search?q="
        + quote(search_query)
    )

    webbrowser.open(
        url
    )


# ============================================================
# SINGER SEARCH
# ============================================================

def search_singer(command):

    singer_name = command

    phrases = [
        "search singer",
        "search singers",
        "find singer",
        "find singers",
        "singer",
        "singers",
        "songs by"
    ]

    for phrase in phrases:

        singer_name = singer_name.replace(
            phrase,
            "",
            1
        ).strip()

    if not singer_name:

        speak(
            "Which singer would you like to search for?"
        )

        return

    search_query = (
        singer_name + " singer songs"
    )

    speak(
        f"Searching for songs by {singer_name}."
    )

    url = (
        "https://www.youtube.com/results?search_query="
        + quote(search_query)
    )

    webbrowser.open(
        url
    )


# ============================================================
# WEATHER
# ============================================================

def get_weather(city):

    if not WEATHER_API_KEY:

        speak(
            "Weather is not configured yet. "
            "Please add your OpenWeatherMap API key."
        )

        return

    try:

        url = (
            "https://api.openweathermap.org/data/2.5/weather"
        )

        params = {
            "q": city,
            "appid": WEATHER_API_KEY,
            "units": "metric"
        }

        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        data = response.json()

        if response.status_code != 200:

            speak(
                "I could not find weather information "
                f"for {city}."
            )

            return

        temperature = data["main"]["temp"]

        feels_like = data["main"]["feels_like"]

        description = data["weather"][0]["description"]

        humidity = data["main"]["humidity"]

        speak(
            f"The weather in {city} is "
            f"{description}. "
            f"The temperature is "
            f"{temperature:.1f} degrees Celsius. "
            f"It feels like "
            f"{feels_like:.1f} degrees. "
            f"Humidity is {humidity} percent."
        )

    except Exception as error:

        print(
            "Weather error:",
            error
        )

        speak(
            "Sorry, I could not get the weather."
        )


def weather_command(command):

    # Examples:
    # weather in Delhi
    # what's the weather in Bhubaneswar

    match = re.search(
        r"(?:weather|temperature|forecast)"
        r".*?"
        r"(?:in|at)\s+(.+)",
        command
    )

    if match:

        city = match.group(1).strip()

    else:

        speak(
            "Which city would you like the weather for?"
        )

        city = listen()

        if not city:

            return

    get_weather(
        city
    )


# ============================================================
# REMINDER
# ============================================================

def reminder_thread(
    seconds,
    message
):

    time.sleep(
        seconds
    )

    speak(
        f"Reminder: {message}"
    )


def set_reminder(command):

    # Supports:
    # remind me in 10 minutes
    # remind me in 1 hour

    match = re.search(
        r"remind me in (\d+)\s*"
        r"(second|seconds|minute|minutes|hour|hours)",
        command
    )

    if not match:

        speak(
            "Please say something like "
            "remind me in 10 minutes."
        )

        return

    amount = int(
        match.group(1)
    )

    unit = match.group(2)

    if "second" in unit:

        seconds = amount

    elif "minute" in unit:

        seconds = amount * 60

    elif "hour" in unit:

        seconds = amount * 3600

    else:

        speak(
            "I could not understand the reminder time."
        )

        return

    # Try to get reminder message
    message = re.sub(
        r"remind me in \d+\s*"
        r"(second|seconds|minute|minutes|hour|hours)",
        "",
        command
    ).strip()

    if not message:

        message = (
            "Your reminder time is up."
        )

    thread = threading.Thread(
        target=reminder_thread,
        args=(
            seconds,
            message
        ),
        daemon=True
    )

    thread.start()

    speak(
        f"Okay. I will remind you in "
        f"{amount} {unit}."
    )


# ============================================================
# EMAIL
# ============================================================

def send_email():

    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:

        speak(
            "Email is not configured. "
            "Please configure your email address "
            "and application password."
        )

        return

    speak(
        "Who should receive the email? "
        "Please enter the email address."
    )

    receiver = input(
        "Receiver email: "
    ).strip()

    if not receiver:

        speak(
            "No receiver was provided."
        )

        return

    speak(
        "What should I write in the email?"
    )

    message_text = input(
        "Email message: "
    ).strip()

    if not message_text:

        speak(
            "The email message is empty."
        )

        return

    email = EmailMessage()

    email["From"] = EMAIL_ADDRESS

    email["To"] = receiver

    email["Subject"] = (
        "Voice Assistant Email"
    )

    email.set_content(
        message_text
    )

    try:

        with smtplib.SMTP(
            "smtp.gmail.com",
            587
        ) as server:

            server.starttls()

            server.login(
                EMAIL_ADDRESS,
                EMAIL_PASSWORD
            )

            server.send_message(
                email
            )

        speak(
            "Your email has been sent successfully."
        )

    except Exception as error:

        print(
            "Email error:",
            error
        )

        speak(
            "I could not send the email. "
            "Please check your email configuration."
        )


# ============================================================
# KNOWLEDGE / WIKIPEDIA
# ============================================================
def knowledge_search(question):
    """Answer general knowledge questions using Wikipedia."""

    if not question:
        speak("What would you like to know?")
        return

    try:
        # Remove common question phrases
        clean_question = question.lower().strip()

        for phrase in [
            "who is ",
            "who was ",
            "what is ",
            "what are ",
            "where is ",
            "when was ",
            "tell me about ",
            "explain "
        ]:
            if clean_question.startswith(phrase):
                clean_question = clean_question[len(phrase):].strip()
                break

        if not clean_question:
            speak("What would you like to know?")
            return

        # Wikipedia search API
        search_url = "https://en.wikipedia.org/w/api.php"

        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": clean_question,
            "srlimit": 1
        }

        headers = {
            "User-Agent": "OIBSIP-VoiceAssistant/1.0"
        }

        response = requests.get(
            search_url,
            params=params,
            headers=headers,
            timeout=10
        )

        print("Wikipedia status:", response.status_code)

        if response.status_code != 200:
            speak("Wikipedia is currently unavailable.")
            return

        data = response.json()

        results = data.get("query", {}).get("search", [])

        if not results:
            speak("I could not find an answer.")
            return

        title = results[0]["title"]

        # Get article summary
        summary_url = (
            "https://en.wikipedia.org/api/rest_v1/page/summary/"
            + quote(title.replace(" ", "_"))
        )

        summary_response = requests.get(
            summary_url,
            headers=headers,
            timeout=10
        )

        print("Wikipedia summary status:",
              summary_response.status_code)

        if summary_response.status_code != 200:
            speak(
                f"I found information about {title}, "
                "but could not retrieve the summary."
            )
            return

        summary_data = summary_response.json()

        extract = summary_data.get("extract", "")

        if not extract:
            speak(
                f"I found {title}, but there is no summary available."
            )
            return

        # Keep response short enough for speech
        if len(extract) > 500:
            extract = extract[:500] + "..."

        speak(extract)

    except requests.exceptions.RequestException as error:
        print("Wikipedia connection error:", error)

        speak(
            "I could not connect to Wikipedia. "
            "Please check your internet connection."
        )

    except ValueError as error:
        print("Wikipedia JSON error:", error)

        speak(
            "Wikipedia returned an invalid response."
        )

    except Exception as error:
        print("Knowledge error:", error)

        speak(
            "Sorry, I could not find the answer."
        )


# ============================================================
# CUSTOM COMMANDS
# ============================================================

CUSTOM_COMMANDS = {

    "open youtube":
        "https://www.youtube.com",

    "open google":
        "https://www.google.com",

    "open github":
        "https://github.com",

    "open gmail":
        "https://mail.google.com"
}


def check_custom_command(command):

    for phrase, url in CUSTOM_COMMANDS.items():

        if phrase in command:

            speak(
                f"Opening "
                f"{phrase.replace('open ', '')}."
            )

            webbrowser.open(
                url
            )

            return True

    return False


# ============================================================
# HELP
# ============================================================

def show_help():

    print("\n")

    print(
        "=" * 60
    )

    print(
        "VOICE ASSISTANT COMMANDS"
    )

    print(
        "=" * 60
    )

    commands = [

        "Hello",

        "What is the time?",

        "What is today's date?",

        "Search Python Django",

        "Play new Odia songs",

        "Play Arijit Singh songs",

        "Play Hindi songs",

        "Search movie Avatar",

        "Search singer Arijit Singh",

        "What's the weather in Delhi?",

        "Remind me in 10 minutes",

        "Send an email",

        "Who is Mahatma Gandhi?",

        "What is Python?",

        "Open YouTube",

        "Open Google",

        "Help",

        "Exit"
    ]

    for command in commands:

        print(
            " -",
            command
        )

    print(
        "=" * 60
    )

    print()


# ============================================================
# TIME
# ============================================================

def tell_time():

    current_time = (
        datetime.datetime.now().strftime(
            "%I:%M %p"
        )
    )

    speak(
        f"The current time is {current_time}."
    )


# ============================================================
# DATE
# ============================================================

def tell_date():

    current_date = (
        datetime.datetime.now().strftime(
            "%B %d, %Y"
        )
    )

    speak(
        f"Today's date is {current_date}."
    )


# ============================================================
# COMMAND HANDLER
# ============================================================

def handle_command(command):

    if not command:

        return True

    command = command.lower().strip()


    # --------------------------------------------------------
    # EXIT
    # --------------------------------------------------------

    if any(
        word in command
        for word in [
            "exit",
            "quit",
            "stop",
            "goodbye",
            "close assistant"
        ]
    ):

        speak(
            "Goodbye! Have a great day."
        )

        return False


    # --------------------------------------------------------
    # HELP
    # --------------------------------------------------------

    if (
        command == "help"
        or command == "commands"
        or command == "what can you do"
    ):

        show_help()

        speak(
            "I can tell time and date, "
            "search the web, play songs, "
            "search movies and singers, "
            "check weather, set reminders, "
            "send emails and answer knowledge questions."
        )

        return True


    # --------------------------------------------------------
    # CUSTOM COMMANDS
    # --------------------------------------------------------

    if check_custom_command(command):

        return True


    # --------------------------------------------------------
    # NLTK INTENT DETECTION
    # --------------------------------------------------------

    intent = detect_intent(
        command
    )

    print(
        "Detected intent:",
        intent
    )


    # --------------------------------------------------------
    # GREETING
    # --------------------------------------------------------

    if intent == "greeting":

        speak(
            "Hello! How can I help you?"
        )

        return True


    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    if intent == "time":

        tell_time()

        return True


    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    if intent == "date":

        tell_date()

        return True


    # --------------------------------------------------------
    # WEATHER
    # --------------------------------------------------------

    if intent == "weather":

        weather_command(
            command
        )

        return True


    # --------------------------------------------------------
    # REMINDER
    # --------------------------------------------------------

    if intent == "reminder":

        set_reminder(
            command
        )

        return True


    # --------------------------------------------------------
    # EMAIL
    # --------------------------------------------------------

    if intent == "email":

        send_email()

        return True


    # --------------------------------------------------------
    # MUSIC
    # --------------------------------------------------------

    if intent == "music":

        play_song(
            command
        )

        return True


    # --------------------------------------------------------
    # SINGER
    # --------------------------------------------------------

    if intent == "singer":

        search_singer(
            command
        )

        return True


    # --------------------------------------------------------
    # MOVIE
    # --------------------------------------------------------

    if intent == "movie":

        search_movie(
            command
        )

        return True


    # --------------------------------------------------------
    # WEB SEARCH
    # --------------------------------------------------------

    if intent == "search":

        query = command

        query = query.replace(
            "search the web",
            ""
        )

        query = query.replace(
            "search the internet",
            ""
        )

        query = query.replace(
            "search online",
            ""
        )

        query = query.replace(
            "search",
            "",
            1
        )

        query = query.replace(
            "google",
            "",
            1
        ).strip()

        # If there is no useful query,
        # ask the user.
        if not query:

            speak(
                "What would you like me to search for?"
            )

            query = listen()

            if not query:

                return True

        web_search(
            query
        )

        return True


    # --------------------------------------------------------
    # KNOWLEDGE QUESTIONS
    # --------------------------------------------------------

    if intent == "knowledge":

        knowledge_search(
            command
        )

        return True


    # --------------------------------------------------------
    # FALLBACK FOR SPECIAL COMMANDS
    # --------------------------------------------------------

    # These checks are kept as a fallback.
    # NLTK handles natural language first,
    # but the old commands still work.

    if re.search(
        r"\b(hello|hi|hey)\b",
        command
    ):

        speak(
            "Hello! How can I help you?"
        )

        return True


    if (
        "what time" in command
        or "current time" in command
        or "tell me the time" in command
        or command == "time"
    ):

        tell_time()

        return True


    if (
        "today's date" in command
        or "todays date" in command
        or "what date" in command
        or "current date" in command
        or command == "date"
    ):

        tell_date()

        return True


    if (
        "weather" in command
        or "temperature" in command
        or "forecast" in command
    ):

        weather_command(
            command
        )

        return True


    if (
        "remind me" in command
        or "set a reminder" in command
    ):

        set_reminder(
            command
        )

        return True


    if (
        "send an email" in command
        or "send email" in command
        or "email someone" in command
    ):

        send_email()

        return True


    if (
        command.startswith("play ")
        or "play song" in command
        or "play music" in command
        or "play new" in command
        or "play latest" in command
    ):

        play_song(
            command
        )

        return True


    if (
        "search singer" in command
        or "find singer" in command
        or "songs by" in command
    ):

        search_singer(
            command
        )

        return True


    if (
        "search movie" in command
        or "find movie" in command
        or "search movies" in command
        or "find movies" in command
    ):

        search_movie(
            command
        )

        return True


    if (
        command.startswith("search ")
        or command.startswith("google ")
        or "search the web" in command
    ):

        query = command

        query = query.replace(
            "search the web",
            ""
        )

        query = query.replace(
            "search",
            "",
            1
        )

        query = query.replace(
            "google",
            "",
            1
        ).strip()

        web_search(
            query
        )

        return True


    # --------------------------------------------------------
    # UNKNOWN COMMAND
    # --------------------------------------------------------

    speak(
        "I don't know that command yet. "
        "Say help to hear the commands I understand."
    )

    return True


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("\n")

    print(
        "=" * 60
    )

    print(
        "        OIBSIP VOICE ASSISTANT"
    )

    print(
        "=" * 60
    )

    print(
        "Task 1 - Beginner + Advanced Features"
    )

    print(
        "=" * 60
    )

    print()

    speak(
        "Voice assistant started."
    )

    speak(
        "You can ask me for the time, date, "
        "weather, songs, movies, singers, "
        "reminders, emails and general knowledge."
    )

    speak(
        "Say help to hear my commands."
    )

    speak(
        "Say exit when you want to stop."
    )

    while True:

        command = listen()

        if command:

            running = handle_command(
                command
            )

            if not running:

                break


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":

    main()