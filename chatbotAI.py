import nltk # type: ignore
import spacy # type: ignore
import random
import datetime
from transformers import pipeline # type: ignore

# Download NLTK resources
nltk.download('punkt')

# Load spaCy model for NLP processing
nlp = spacy.load('en_core_web_sm')

# Load a pre-trained transformer model for text generation (e.g., GPT-2)
chatbot_model = pipeline('text-generation', model='gpt2')

def preprocess_text(text):
    # Tokenize using spaCy
    doc = nlp(text.lower())
    # Filter out stop words and punctuations, then lemmatize
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

def classify_intent(user_input):
    user_input = user_input.lower()

    if "weather" in user_input:
        return "weather"
    elif "name" in user_input:
        return "name"
    elif "time" in user_input:
        return "time"
    else:
        return "general"

def generate_response(user_input):
    intent = classify_intent(user_input)

    if intent == "weather":
        return "I can't get real-time data yet, but it looks sunny!"
    elif intent == "name":
        return "My name is AI-Chatbot, your virtual assistant."
    elif intent == "time":
        now = datetime.datetime.now()
        return f"The current time is {now.strftime('%H:%M:%S')}."
    else:
        # Use GPT-2 to generate a response for more general queries
        gpt2_response = chatbot_model(user_input, max_length=50, num_return_sequences=1)
        return gpt2_response[0]['generated_text']

def chatbot():
    print("Hello! I'm your AI-powered chatbot. Type 'exit' to end the chat.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        # Preprocess user input
        processed_input = preprocess_text(user_input)
        # Generate and print the response
        response = generate_response(processed_input)
        print(f"Bot: {response}")

# Start the chatbot
chatbot()