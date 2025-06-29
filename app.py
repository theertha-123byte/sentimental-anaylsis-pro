import joblib
import gradio as gr

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def predict_sentiment(text):
    text_transformed = vectorizer.transform([text])
    prediction = model.predict(text_transformed)[0]
    return "Positive 😊" if prediction == 1 else "Negative 😞"

demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Type your review here..."),
    outputs="text",
    title="IMDB Sentiment Analysis",
    description="Enter a movie review and find out if it's Positive or Negative!"
)

demo.launch()
