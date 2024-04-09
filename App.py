import streamlit as st
import pickle
import nltk


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

from nltk.corpus import stopwords

import string

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    z = []
    for i in y:
        if i not in stopwords.words('english') and i not in string.punctuation:
            z.append(i)
    x = []
    for i in z:
        x.append(ps.stem(i))

    return " ".join(x)  # joining all the list item with space between them


tfidf = pickle.load(open('Vectorizer.pkl', 'rb'))
Model = pickle.load(open('Model.pkl', 'rb'))


st.title('SMS-Spam-Classifier')

# text_input
text_input = st.text_area('Enter Your Message Here')

# if button is pressed then perform these operations and predict
if st.button('predict'):
    # transform_text
    Transformed_text = transform_text(text_input)

    # Vectorize
    Vectorized_text = tfidf.transform([Transformed_text])
    # Model to predict
    prediction = Model.predict(Vectorized_text)[0]
    # display the result

    if prediction == 0:
        st.write('Hem')
    else :
        st.write('Spam')

