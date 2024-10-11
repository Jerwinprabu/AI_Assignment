import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('all')
nltk.download('wordnet')

# Task 1: Content Classifier using Decision Tree
def content_classifier():
    """Classifies webtoon descriptions into categories using Decision Tree."""
    print("\n=== Task 1: Content Classifier ===")
    
    try:
        # Load CSV data for webtoon descriptions
        data = pd.read_csv('webtoon_descriptions.csv')
        if data.empty:
            print("Error: The 'webtoon_descriptions.csv' file is empty.")
            return

        descriptions = data['description']
        labels = data['category']
        
        # Text preprocessing and vectorization
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(descriptions)
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test, desc_train, desc_test = train_test_split(
            X, labels, descriptions, test_size=0.33, random_state=42
        )
        
        # Using a Decision Tree Classifier
        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        
        # Predict on the full dataset (training + testing)
        predictions_train = clf.predict(X_train)
        predictions_test = clf.predict(X_test)
        
        # Combine train and test sets for full predictions
        full_predictions = list(predictions_train) + list(predictions_test)
        full_descriptions = list(desc_train) + list(desc_test)
        
        # Print all descriptions and their predicted categories
        print(f"Number of Descriptions: {len(full_descriptions)}")
        for desc, pred in zip(full_descriptions, full_predictions):
            print(f"Description: {desc[:50]}... | Predicted Category: {pred}")
    
    except FileNotFoundError:
        print("Error: 'webtoon_descriptions.csv' file not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Task 2: Sentiment Analysis using TextBlob
def sentiment_analysis():
    """Analyzes sentiment of user comments."""
    print("\n=== Task 2: Sentiment Analysis ===")
    
    try:
        # Load CSV data for user comments
        data = pd.read_csv('comments.csv')
        if data.empty:
            print("Error: The 'comments.csv' file is empty.")
            return

        comments = data['comment']
        
        # Analyzing sentiment of comments for all comments
        positive_comments = 0
        sentiment_results = []  # Store results in a list for structured output

        for comment in comments:
            analysis = TextBlob(comment)
            polarity = analysis.sentiment.polarity
            sentiment_results.append((comment[:50], polarity))  # Store a snippet of comment and its polarity
            if polarity > 0:
                positive_comments += 1
        
        # Display results
        for comment_snippet, polarity in sentiment_results:
            print(f"Comment: {comment_snippet}... | Sentiment Polarity: {polarity}")
        
        # Calculating percentage of positive comments
        total_comments = len(comments)
        positive_percentage = (positive_comments / total_comments) * 100
        print(f"\nPositive comments: {positive_percentage:.2f}%")
    
    except FileNotFoundError:
        print("Error: 'comments.csv' file not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Task 3: Chatbot using NLTK for basic NLP
def chatbot():
    print("\n=== Task 3: Chatbot ===")
    
    try:
        # Load chatbot questions and responses from CSV
        data = pd.read_csv('chatbot_data.csv')
        questions = data['question']
        responses = data['response']
        
        # Debug: Print loaded questions and responses
        print("Loaded Questions:")
        print(questions)
        print("Loaded Responses:")
        print(responses)

        # Initialize NLTK lemmatizer for normalization
        lemmatizer = WordNetLemmatizer()

        # Function to preprocess user input
        def preprocess_input(user_input):
            # Tokenize the input
            tokens = word_tokenize(user_input.lower())
            # Lemmatize the tokens
            lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
            return lemmatized_tokens

        # Function to match user input with chatbot's keywords
        def chatbot_response(user_input):
            processed_input = preprocess_input(user_input)
            print(f"Processed User Input: {processed_input}")  # Debugging input
            
            for question, response in zip(questions, responses):
                # Preprocess the question from the dataset
                processed_question = preprocess_input(question)
                print(f"Processed Question: {processed_question}")  # Debugging question
                
                # If the keywords from the user input match the keywords from the question, return the response
                if set(processed_question).intersection(processed_input):
                    return response
            
            return "Sorry, I don't have an answer for that."

        # Simulating a conversation until the user types "exit"
        while True:
            user_question = input("User: ")
            if user_question.lower() in ['exit', 'quit', 'bye']:
                print("Chatbot: Goodbye!")
                break
            print(f"Chatbot: {chatbot_response(user_question)}")

    except FileNotFoundError:
        print("Error: 'chatbot_data.csv' file not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Main function to run all tasks
def main():
    content_classifier()
    sentiment_analysis()
    chatbot()

# Execute all tasks
if __name__ == "__main__":
    main()
