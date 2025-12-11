# **AI Emotion Predictor**

A minimal NLP project that predicts the emotion of a given text using a fine-tuned **DistilBERT** model.

---

**Requirements**
- Python 3.10+ (recommended)

**Dependencies**
- torch
- transformers
- scikit-learn
- pandas


## **Project Structure**

### **1. `trainer.py`**
- Loads the dataset in `text;label` format  
- Fine-tunes a DistilBERT sequence-classification model  
- Encodes labels using **LabelEncoder**  
- Saves the trained model, tokenizer, and label encoder to `model_output/`

---

### **2. `predictor.py`**
- Loads the fine-tuned model and tokenizer  
- Loads the saved **LabelEncoder**  
- Tokenizes input text, runs inference, and returns the predicted emotion  

---

### **3. `chat.py`**
- Console-based chatbot  
- Takes user input  
- Passes it to the predictor and prints the detected emotion  

---

## **How to Run**

### **1. Install dependencies:**
```bash
pip install -r requirements.txt
```

### **2. Train the model**
(This generates the model_output/ directory containing the trained components)
```bash
python model/trainer.py
```
### **3. Run the chatbot**
```bash
python app/chat.py
```

**How the Prediction Pipeline Works:**

When you type a sentence: // eg.-> _you: today my boss shouted at me._

chat.py sends your sentence to EmotionPredictor.

The predictor tokenizes the text using the saved tokenizer.

The "DistilBERT" model processes the tokens and outputs logits.

Softmax converts logits into probabilities.

The index with the highest probability is mapped back using LabelEncoder.

The decoded label is printed as the predicted emotion. // expected answer:-> _Bot: The detected emotion is **anger**_


###**Note:**###
This chatbot is not perfect and may make mistakes. This is mainly due to the small size of the dataset and limited data coverage, which prevented the model from being fully fine-tuned.

**In future iterations, the system will be improved to:**

- Better predict emotions with more comprehensive data

- Incorporate more nuanced emotion detection

- Possibly detect emotions from both text and voice

_The dataset used for training was taken from Kaggle: Emotions Dataset for NLP_
