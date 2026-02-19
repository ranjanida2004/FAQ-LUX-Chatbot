import gradio as gr
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load FAQ data
with open("product_faq_500.json", "r") as file:
    faq_data = json.load(file)

questions = list(faq_data.keys())
answers = list(faq_data.values())

# Vectorize questions
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

def chatbot_response(user_input):
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)
    best_match_index = similarity.argmax()
    best_score = similarity[0][best_match_index]

    if best_score < 0.3:
        return "Sorry, I couldn't understand your question."

    return answers[best_match_index]

# Gradio interface
interface = gr.Interface(
    fn=chatbot_response,
    inputs=gr.Textbox(label="Ask your question"),
    outputs="text",
    title="FAQ LUX Chatbot",
    description="Ask any question related to our services."
)

if __name__ == "__main__":
    interface.launch()
