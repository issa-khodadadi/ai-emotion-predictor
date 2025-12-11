import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.predictor import EmotionPredictor

def chat():
    bot = EmotionPredictor()
    print("Hello! Type your sentence and I will predict the emotion. (type 'exit' to quit)")

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "exit":
            break
        emotion = bot.predict(user_input)
        print(f"Bot: The detected emotion is **{emotion}**")

if __name__ == "__main__":
    chat()
