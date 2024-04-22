
from flask import Flask, render_template, request, jsonify
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the Flask app
app = Flask(__name__, static_url_path='/Static')

# Load your dataset
data = pd.read_excel('add your dataset')

# Combine relevant columns into one text
data['combined_text'] = data['job_description'] + ' ' + data['work_environment'] + ' ' + data['qualification'] 

# Convert the combined text to lowercase
data['combined_text'] = data['combined_text'].str.lower()

# Tokenize the text data
data['tokens'] = data['combined_text'].apply(word_tokenize)

# Remove punctuation and stopwords
stop_words = set(stopwords.words('english'))

def preprocess_text(tokens):
    # Remove punctuation
    tokens = [word for word in tokens if word.isalnum()]

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens

data['preprocessed_tokens'] = data['tokens'].apply(preprocess_text)

# Create TF-IDF vectors
tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=True)
tfidf_matrix = tfidf_vectorizer.fit_transform(data['preprocessed_tokens'].apply(lambda x: ' '.join(x)))

# Define a function to recommend careers based on user input
def recommend_careers(user_input, tfidf_matrix, data):
    # Preprocess user input
    user_input = user_input.lower()
    user_tokens = word_tokenize(user_input)
    user_tokens = preprocess_text(user_tokens)

    # Transform user input into a TF-IDF vector
    user_input_vector = tfidf_vectorizer.transform([' '.join(user_tokens)])

    # Calculate cosine similarities between user input and careers
    cosine_similarities = cosine_similarity(user_input_vector, tfidf_matrix)

    # Get career indices sorted by similarity
    career_indices = cosine_similarities.argsort()[0][::-1]

    # Get recommended careers
    recommended_careers = data['job_name'].iloc[career_indices]

    return recommended_careers

# Serve the HTML file
@app.route('/')
def index():
    return render_template('index.html')

# Define the API endpoint for career recommendations
@app.route('/recommend_careers', methods=['POST'])
def api_recommend_careers():
    request_data = request.get_json()
    job_description = request_data.get('job_description', '')
    work_environment = request_data.get('work_environment', '')
    qualification = request_data.get('qualification', '')

    # Combine the individual inputs into a single user input string
    user_input = f"{job_description} {work_environment} {qualification}"

    # Check if all input fields are empty
    if not any(user_input.split()):
        return jsonify({"recommended_careers": []})  # Return an empty list

    recommended_careers = recommend_careers(user_input, tfidf_matrix, data)

    # Log recommendations for debugging
    print(recommended_careers.head(5).tolist())

    return jsonify({"recommended_careers": recommended_careers.head(5).tolist()})

if __name__ == '__main__':
    app.run(debug=True)

