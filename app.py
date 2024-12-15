from flask import Flask, request,redirect,url_for, render_template
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, cross_val_score
from statistics import mean
from nltk.corpus import wordnet 
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from itertools import combinations
from time import time
from collections import Counter
import operator
#from xgboost import XGBClassifier
import math
import pickle
import openpyxl
from Treatment import diseaseDetail
from sklearn.linear_model import LogisticRegression
from flask import session

# Suppress warnings (to make the output cleaner)
warnings.simplefilter("ignore")


app=Flask(__name__,static_url_path='/static')
app.secret_key = 'super secret key'    

# Global variable to store selected symptoms during the process
global select_list

def synonyms(term):
    """
    Function to find synonyms of a given term using both web scraping and the WordNet lexical database.

    Parameters:
    term (str): The word for which synonyms are to be found.

    Returns:
    set: A set of unique synonyms for the input term.
    """
    synonyms = []  # Initialize an empty list to store synonyms

    # Step 1: Web scraping synonyms from Thesaurus.com
    response = requests.get('https://www.thesaurus.com/browse/{}'.format(term))  # Send an HTTP GET request to Thesaurus.com
    soup = BeautifulSoup(response.content, "html.parser")  # Parse the HTML response using BeautifulSoup

    try:
        # Locate the section containing synonyms in the Thesaurus.com page
        container = soup.find('section', {'class': 'MainContentContainer'})  # Find the main content container
        row = container.find('div', {'class': 'css-191l5o0-ClassicContentCard'})  # Find the specific synonym card
        row = row.find_all('li')  # Extract all <li> elements (each contains a synonym)

        # Extract the text of each <li> element and add it to the synonyms list
        for x in row:
            synonyms.append(x.get_text())
    except:
        # Handle any errors during the web scraping process (e.g., if the structure of the page changes)
        None

    # Step 2: Adding synonyms from WordNet
    for syn in wordnet.synsets(term):  # Get all synsets (word senses) for the term from WordNet
        synonyms += syn.lemma_names()  # Add the lemma names (synonyms) for each synset to the list

    # Step 3: Return a set of unique synonyms to avoid duplicates
    return set(synonyms)


def similarity(dataset_symptoms, user_symptoms):
    """
    Function to find matching symptoms from a dataset based on similarity with user-provided symptoms.

    Parameters:
    dataset_symptoms (list of str): List of symptoms from the dataset.
    user_symptoms (list of str): List of symptoms provided by the user.

    Returns:
    list: A list of symptoms from the dataset that are sufficiently similar to user-provided symptoms.
    """
    found_symptoms = set()  # Initialize an empty set to store matched symptoms (to avoid duplicates)

    # Loop through each symptom in the dataset
    for idx, data_sym in enumerate(dataset_symptoms):
        # Split the dataset symptom into individual words
        data_sym_split = data_sym.split()

        # Compare the current dataset symptom with each user-provided symptom
        for user_sym in user_symptoms:
            count = 0  # Initialize a counter to track matching words

            # Check each word in the dataset symptom
            for symp in data_sym_split:
                # If the word exists in the user symptom, increase the count
                if symp in user_sym.split():
                    count += 1

            # Calculate the similarity by dividing the count by the length of the dataset symptom
            # If more than 50% of the words match, consider it similar
            if count / len(data_sym_split) > 0.5:
                found_symptoms.add(data_sym)  # Add the matched symptom to the set

    # Convert the set of found symptoms to a list (to preserve a consistent return type)
    found_symptoms = list(found_symptoms)
    return found_symptoms  # Return the list of matched symptoms
 
def preprocess(user_symptoms):
    """
    Function to preprocess user-provided symptoms for comparison with a disease-symptom dataset.

    Parameters:
    user_symptoms (list of str): A list of symptoms provided by the user.

    Returns:
    list of str: Preprocessed symptoms that include possible synonyms and combinations.
    """
    # Load the datasets:
    # df_comb contains disease combinations and their associated symptoms.
    # df_norm contains normalized data of individual diseases and symptoms.
    df_comb = pd.read_csv(r"Dataset/dis_sym_dataset_comb.csv")  # Dataset for disease combinations
    df_norm = pd.read_csv(r"Dataset/dis_sym_dataset_norm.csv")  # Dataset for individual diseases

    # Extract features (symptoms) and labels (diseases) from the combination dataset
    X = df_comb.iloc[:, 1:]  # Symptoms (all columns except the first)
    Y = df_comb.iloc[:, 0:1]  # Diseases (first column)

    # Get the list of symptoms from the dataset (used later for matching)
    dataset_symptoms = list(X.columns)

    # Initialize stop words, lemmatizer, and tokenizer
    stop_words = stopwords.words('english')  # Common stop words in English
    lemmatizer = WordNetLemmatizer()  # Lemmatizer for reducing words to their base forms
    splitter = RegexpTokenizer(r'\w+')  # Tokenizer to split text into words (ignoring punctuation)

    processed_user_symptoms = []  # Initialize a list to store processed user symptoms

    # Step 1: Clean and lemmatize user symptoms
    for sym in user_symptoms:
        sym = sym.strip()  # Remove leading and trailing whitespace
        sym = sym.replace('-', ' ')  # Replace hyphens with spaces
        sym = sym.replace("'", '')  # Remove apostrophes
        # Lemmatize and tokenize the symptom
        sym = ' '.join([lemmatizer.lemmatize(word) for word in splitter.tokenize(sym)])
        processed_user_symptoms.append(sym)  # Append the cleaned symptom to the list

    user_symptoms = []  # Initialize a new list to store final user symptoms

    # Step 2: Generate combinations and synonyms for each processed symptom
    for user_sym in processed_user_symptoms:
        user_sym = user_sym.split()  # Split the symptom into words
        str_sym = set()  # Use a set to store unique synonyms and combinations

        # Generate all possible word combinations of the symptom
        for comb in range(1, len(user_sym) + 1):
            for subset in combinations(user_sym, comb):  # Generate subsets of words
                subset = ' '.join(subset)  # Join the subset into a string
                subset = synonyms(subset)  # Get synonyms for the subset
                str_sym.update(subset)  # Add the synonyms to the set

        # Add the original symptom (as a single string) to the set
        str_sym.add(' '.join(user_sym))

        # Combine all synonyms into a single string, replace underscores, and append to the list
        user_symptoms.append(' '.join(str_sym).replace('_', ' '))

    # Return the fully processed list of user symptoms
    return user_symptoms

# Define the root route ("/") that supports both POST and GET methods
@app.route("/", methods=["POST", "GET"])
def index():
    """
    Handles requests to the root URL ("/").
    Renders and returns the 'index.html' template.
    """
    return render_template("index.html")  # Render the main index page

# Define the root route ("/index") that supports both POST and GET methods
@app.route("/index", methods=["POST", "GET"])
def index1():
    """
    Handles requests to the root URL ("/").
    Renders and returns the 'index.html' template.
    """
    return render_template("index.html")  # Render the main index page

# Define the "/about" route that supports both POST and GET methods
@app.route("/about", methods=["POST", "GET"])
def about():
    """
    Handles requests to the '/about' URL.
    Renders and returns the 'about.html' template.
    """
    return render_template("about.html")  # Render the about page

# Define the "/demo" route that supports only GET requests (default behavior)
@app.route("/demo")
def demo():
    """
    Handles requests to the '/demo' URL.
    Renders and returns the 'demo.html' template.
    """
    return render_template('demo.html')  # Render the demo page

@app.route("/predict", methods=["POST", "GET"])
def predict():
    """
    Handles requests to the '/predict' route.
    - Processes user-submitted symptoms.
    - Matches the symptoms against a dataset.
    - Predicts potential diseases based on the matched symptoms.
    - Renders the 'predict.html' template with results.
    """

    # Load datasets for disease-symptom combinations and individual diseases
    df_comb = pd.read_csv(r"Dataset/dis_sym_dataset_comb.csv")  # Disease combination dataset
    df_norm = pd.read_csv(r"Dataset/dis_sym_dataset_norm.csv")  # Individual disease dataset

    # Separate symptoms and diseases columns from the combination dataset
    X = df_comb.iloc[:, 1:]  # Symptoms columns
    Y = df_comb.iloc[:, 0:1]  # Disease column
    dataset_symptoms = list(X.columns)  # List of all symptoms in the dataset

    found_symptoms = set()  # Set to store matched symptoms

    # Retrieve user-submitted symptoms from the form (comma-separated input)
    user_symptoms = list(request.form.get('symptoms', 'False').split(','))
    print(user_symptoms)  # Debug: Print user-provided symptoms

    # Preprocess user symptoms (lemmatization, synonyms, etc.)
    user_symptoms = preprocess(user_symptoms)

    # Find matching symptoms from the dataset based on user input
    found_symptoms = similarity(dataset_symptoms, user_symptoms)
    print(found_symptoms)  # Debug: Print matched symptoms

    select_list = []  # List to store indices of matched symptoms
    print("Top matching symptoms from your search!")
    for idx, symp in enumerate(found_symptoms):
        select_list.append(idx)  # Add index of each found symptom
    dis_list = set()  # Set to store possible diseases based on symptoms
    print(select_list)  # Debug: Print indices of matched symptoms

    # Variables to store final matched symptoms and additional symptoms for prediction
    final_symp = []
    counter_list = []

    # Identify diseases associated with the matched symptoms
    for idx in select_list:
        symp = found_symptoms[int(idx)]  # Retrieve symptom by index
        final_symp.append(symp)  # Add to final symptom list
        # Find diseases where this symptom appears in the individual disease dataset
        dis_list.update(set(df_norm[df_norm[symp] == 1]['label_dis']))

    # Count occurrences of other symptoms associated with the potential diseases
    for dis in dis_list:
        row = df_norm.loc[df_norm['label_dis'] == dis].values.tolist()  # Get disease row
        row[0].pop(0)  # Remove the disease name (first column)
        for idx, val in enumerate(row[0]):
            if val != 0 and dataset_symptoms[idx] not in final_symp:
                counter_list.append(dataset_symptoms[idx])  # Count symptoms not in final_symp

    # Create a dictionary to count occurrences of additional symptoms
    dict_symp = dict(Counter(counter_list))
    # Sort the dictionary by symptom frequency (descending order)
    dict_symp_tup = sorted(dict_symp.items(), key=operator.itemgetter(1), reverse=True)
    print("dictionary:", dict_symp_tup)  # Debug: Print sorted symptom frequency dictionary

    # Generate a list of additional symptoms for suggestions
    another_symptoms = []
    count = 0  # Counter for the number of additional symptoms
    for tup in dict_symp_tup:
        count += 1
        another_symptoms.append(tup[0])  # Add symptom to the suggestions list

    # Store variables in the session for use in subsequent requests
    session['my_var'] = another_symptoms  # Additional symptoms
    session['my_var2'] = final_symp  # Matched symptoms
    session['count'] = count  # Count of additional symptoms
    session['dict_symp_tup'] = dict_symp_tup  # Sorted dictionary of additional symptoms
    session['tup'] = tup  # Last processed tuple (for debugging or further use)

    # Render the 'predict.html' template with relevant data
    return render_template(
        "predict.html",
        found_symptoms=enumerate(found_symptoms),  # Matched symptoms with indices
        another_symptoms=enumerate(another_symptoms),  # Additional symptoms with indices
        count=count,  # Total number of additional symptoms
        dict_symp_tup=len(dict_symp_tup)  # Number of unique symptoms in the dictionary
    )

@app.route("/next", methods=["POST", "GET"])
def next():
    """
    Handles requests to the '/next' route.
    - Retrieves session data for symptoms and other variables.
    - Processes additional relevant symptoms selected by the user.
    - Updates the sample vector for symptoms and stores it in the session.
    - Renders the 'next.html' template with updated data.
    """

    # Retrieve session variables
    my_var = session.get('my_var', None)  # Additional symptoms suggested earlier
    my_var2 = session.get('my_var2', None)  # Matched symptoms from the previous step
    count = session.get('count', None)  # Count of additional symptoms
    x = session.get('tup', None)  # Last processed tuple (from '/predict' route)

    # Load datasets for disease-symptom combinations and individual diseases
    df_comb = pd.read_csv(r"Dataset/dis_sym_dataset_comb.csv")  # Disease combination dataset
    df_norm = pd.read_csv(r"Dataset/dis_sym_dataset_norm.csv")  # Individual disease dataset

    # Separate symptoms and diseases columns from the combination dataset
    X = df_comb.iloc[:, 1:]  # Symptoms columns
    Y = df_comb.iloc[:, 0:1]  # Disease column
    dataset_symptoms = list(X.columns)  # List of all symptoms in the dataset

    # Retrieve user-selected additional symptoms from the form
    final_symptoms = list(request.form.get('relevance', 'False').split(','))

    # Initialize a vector to represent the presence of symptoms
    sample_x = [0 for x in range(0, len(dataset_symptoms))]  # Initialize a zero vector

    # Update the vector for selected symptoms
    for i in final_symptoms:
        my_var2.append(i)  # Add selected symptoms to the matched symptoms list
        sample_x[dataset_symptoms.index(i)] = 1  # Mark the symptom as present in the vector

    # Store the updated vector in the session
    session['sample_x'] = sample_x
    print("sample_x: ", sample_x)  # Debug: Print the updated symptom vector

    # Render the 'next.html' template with the updated matched symptoms
    return render_template("next.html", my_var2=enumerate(my_var2))  # Pass enumerated matched symptoms to the template

@app.route("/final", methods=["POST", "GET"])
def final():
    """
    Handles the '/final' route.
    - Retrieves the user's symptom vector and other session data.
    - Loads the trained model to predict the top diseases based on the provided symptoms.
    - Calculates probabilities and ranks the diseases.
    - Passes the top predictions to the 'final.html' template for display.
    """

    # Retrieve the symptom vector (generated in the '/next' route) from the session
    sample_x = session.get('sample_x')
    print("Sample symptom vector (sample_x):", sample_x)  # Debug: Print the symptom vector

    # Load datasets containing disease-symptom mappings
    df_comb = pd.read_csv(r"Dataset/dis_sym_dataset_comb.csv")  # Disease combination dataset
    df_norm = pd.read_csv(r"Dataset/dis_sym_dataset_norm.csv")  # Individual disease dataset

    # Extract symptoms and diseases from the datasets
    X = df_comb.iloc[:, 1:]  # Symptoms
    Y = df_comb.iloc[:, 0:1]  # Diseases
    dataset_symptoms = list(X.columns)  # List of symptoms from the dataset

    # Retrieve the user's selected symptoms from the session
    my_var2 = session.get('my_var2')
    print("Final symptoms selected by the user:", my_var2)  # Debug: Print selected symptoms

    # Load the pre-trained model (assumes a saved model file named 'model_saved')
    my_model = pickle.load(open(r'model_saved', 'rb'))

    # Predict probabilities for each disease based on the user's symptoms
    output = my_model.predict_proba([sample_x])

    # Perform cross-validation on the model for performance evaluation
    scores = cross_val_score(my_model, X, Y, cv=10)  # 10-fold cross-validation

    # Define the number of top predictions to show (k = 5)
    k = 5

    # Retrieve the list of unique diseases from the dataset and sort them
    diseases = list(set(Y['label_dis']))
    diseases.sort()

    # Get the indices of the top `k` predictions based on probabilities
    topk = output[0].argsort()[-k:][::-1]
    print(f"\nTop {k} diseases predicted based on symptoms")  # Debug: Log top diseases

    # Initialize a dictionary to store probabilities for the top diseases
    topk_dict = {}

    # Iterate over the top `k` predicted diseases to calculate probabilities
    for idx, t in enumerate(topk):
        match_sym = set()  # Set to store symptoms matching the current disease

        # Retrieve the row corresponding to the disease `t` from the individual disease dataset
        row = df_norm.loc[df_norm['label_dis'] == diseases[t]].values.tolist()
        row[0].pop(0)  # Remove the disease label column

        # Identify symptoms associated with the disease
        for idx, val in enumerate(row[0]):
            if val != 0:  # Check if the symptom is present for the disease
                match_sym.add(dataset_symptoms[idx])

        # Calculate a weighted probability considering matched symptoms and cross-validation score
        prob = (len(match_sym.intersection(set(my_var2))) + 1) / (len(set(my_var2)) + 1)
        prob *= mean(scores)  # Scale probability using the model's cross-validation score
        topk_dict[t] = prob  # Store the probability for the disease

    # Sort the diseases by their calculated probabilities in descending order
    topk_sorted = dict(sorted(topk_dict.items(), key=lambda kv: kv[1], reverse=True))
    print("Top k diseases sorted by probability:", topk_sorted)  # Debug: Log sorted diseases

    # Prepare the results for display
    arr = []  # List to store disease names and probabilities
    for key in topk_sorted:
        prob = topk_sorted[key] * 100  # Convert probability to a percentage
        arr.append(f'Disease name: {diseases[key]}')  # Append the disease name to the results

    # Render the 'final.html' template with the top predictions
    return render_template("final.html", arr=arr)


@app.route("/treatment", methods=["POST", "GET"])
def treatment():
    """
    Handles the '/treatment' route.
    - Retrieves the disease selected by the user.
    - Loads an Excel file containing treatments for diseases.
    - Searches for the disease in the Excel sheet and retrieves the treatment options.
    - Displays the treatment options on the 'treatment.html' template.
    """

    # Get the disease selected by the user from the form (POST request)
    treat_dis = request.form.get('dis', 'False')

    # Load the Excel workbook containing disease treatments
    workbook = openpyxl.load_workbook(r'cure_minor.xlsx')

    # Select the specific worksheet containing the treatment data (assumes sheet name 'Sheet1')
    worksheet = workbook['Sheet1']

    # Initialize an empty list to store treatment options
    ans = []

    # Iterate through each row of the worksheet
    for row in worksheet.iter_rows(values_only=True):
        print("Row data:", row)  # Debug: Print the current row from the worksheet
        print("Disease to treat:", treat_dis)  # Debug: Print the disease selected by the user

        # Check if the selected disease (treat_dis) is in the current row (first column should be the disease name)
        if treat_dis in row:
            print("Disease found in row")  # Debug: Disease found in the row
            # Join all treatment options (from the second column onwards) into a single string
            stri = ''.join(row[1:])
            # Split the treatments by commas to get them as a list
            ans = stri.split(',')
            print("Treatment options:", ans)  # Debug: Print the treatment options found

    # Render the 'treatment.html' template and pass the treatment options (ans) to it
    return render_template("treatment.html", ans=ans)


if __name__ == '__main__':
    """
    This block ensures that the Flask app runs only when the script is executed directly,
    and not when it's imported as a module.
    """
    app.run(
        debug=True,         # Enables Flask's debug mode, which provides detailed error messages and auto-reloading.
        host="0.0.0.0",     # Makes the server accessible on all network interfaces (useful for testing on different devices).
        port=5000,           # Sets the port for the Flask app to run on (default: 5000).
        threaded=True        # Enables multi-threading, allowing multiple requests to be processed simultaneously.
    )
