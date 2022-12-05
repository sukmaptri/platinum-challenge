##import library
from flask import Flask, jsonify, request, make_response, render_template
from flasgger import Swagger, LazyJSONEncoder, LazyString, swag_from

import re
import string
import pandas as pd
import numpy as np
from keras import preprocessing
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import load_model
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords as stopwords_scratch
from nltk.tokenize import word_tokenize

app = Flask(__name__)
app.json_encoder = LazyJSONEncoder
swagger_template = dict(
info = {
    'title': LazyString(lambda: 'API Documentation for Machine Learning'),
    'version': LazyString(lambda: '1.0.0'),
    'description': LazyString(lambda: 'API Documentation for Binar Machine Learning Challenge - Kelompok 3 (Sukma, Zael, Gun)')
    },
    host = LazyString(lambda: request.host)
)
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'docs',
            "route": '/docs.json'
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/"
}

swagger = Swagger(app, template=swagger_template, 
                  config=swagger_config)

max_features = 100000
tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)

sentiment = ['negative', 'neutral', 'positive']

##cleansing process
def remove_emoji(text):
  emoji_pattern = re.compile("["
              u"\U0001F600-\U0001F64F" #emojis
              u"\U0001F300-\U0001F5FF" #symbols & pictographs
              u"\U0001F680-\U0001F6FF" #transport & map symbols
              u"\U0001F1E0-\U0001F1FF" #flags
              u"\U00002702-\U000027B0" 
              u"\U000024C2-\U0001D251"
              "]+", flags=re.UNICODE)
  return emoji_pattern.sub(r"",text)

def remove_url(text):
  url_pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
  return url_pattern.sub(r"",text)

def clean_text(text):
  delete_dict = {sp_character:'' for sp_character in string.punctuation}
  delete_dict[' '] = ' '
  table = str.maketrans(delete_dict)
  text1 = text.translate(table)
  textArr = text1.split()
  text2 = ' '.join([w for w in textArr if (not w.isdigit() and (not w.isdigit() and len(w)>3 ))])
  return text2.lower()

def clean_all(text):
  filter1 = re.sub('([^\x00-\x7f])|(USER)|(@[A-Za-z0-9_]+)|(#[A-Za-z0-9_]+)|(RT)','',text)
  filter2 = re.sub(r'([^\w\s])|(\n)',' ',filter1)
  filter3 = re.sub('&amp;','dan',filter2)
  filter4 = re.sub(' +',' ',filter3)
  filter5 = re.sub("\S*\d\S*", " ", filter4)
  return filter5.lower()

##remove stopwords
def word_best(text):
  my_stop_words = stopwords_scratch.words('indonesian')
  my_stop_words.extend(['nya'])
  my_stop_words.remove('tidak')
  my_stop_words.remove('jangan')

  text_tokens = word_tokenize(text)
  tokens_without_sw = [word for word in text_tokens if not word in my_stop_words]
  for i in tokens_without_sw:
    if len(i)<=3:
      tokens_without_sw.remove(i)
  filtered_sentence = (" ").join(tokens_without_sw)
  return filtered_sentence.lower()

#####NEURAL NETWORK
##open model NN
file = open('C:/PENDIDIKAN/BINAR ACADEMY_DATA SCIENCE/Platinum/cl_platinum/resources_of_nn/x_pad_sequences_nn.pickle', 'rb')
feature_file_from_nn = pickle.load(file)
file.close()

model_file_from_nn = load_model('C:/PENDIDIKAN/BINAR ACADEMY_DATA SCIENCE/Platinum/cl_platinum/ModelNN/modelNN.h5')

##NN input teks
@swag_from("docs/NNForm.yml", methods=['POST'])
@app.route('/NNForm', methods=['POST'])

def NN():

    original_text = request.form.get('text')
    text1 = remove_emoji(original_text)
    text2 = remove_url(text1)
    text3 = clean_text(text2)
    text4 = clean_all(text3)
    text = word_best(text4)
    
    tokenizer.fit_on_texts(text)
    feature = tokenizer.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen = feature_file_from_nn.shape[1])
    prediction = model_file_from_nn.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]

    json_response = {
        'status_code': 200,
        'description': "Neural Network Result of Sentiment Analysis",
        'data': {
            'text': original_text,
            'sentiment': get_sentiment
        },
    }
    response_data = jsonify(json_response)
    return response_data

##NN upload as file
@swag_from("docs/NNFile.yml", methods=['POST'])
@app.route('/NNFile', methods=['POST'])

def NN_FILE():

    file = request.files.getlist('file')[0]
    df = pd.read_csv(file, names = ['text'], header = None)
    texts = df.text.to_list()

    text_with_sentiment = []
    for original_text in texts:
      text1 = remove_emoji(original_text)
      text2 = remove_url(text1)
      text3 = clean_text(text2)
      text4 = clean_all(text3)
      text_clean = word_best(text4)
      
      text_with_sentiment.append(text_clean)
      
      tokenizer.fit_on_texts(text_clean)
      feature = tokenizer.texts_to_sequences(text_clean)
      feature = pad_sequences(feature, maxlen = feature_file_from_nn.shape[1])
      prediction = model_file_from_nn.predict(feature)
      get_sentiment = sentiment[np.argmax(prediction[0])]

      text_with_sentiment.append({
        'text':original_text,
        'sentiment': get_sentiment
      })

    json_response = {
      'status_code': 200,
      'description': "Neural Network Result of Sentiment Analysis",
      'data': text_with_sentiment,
    }
    response_data = jsonify(json_response)
    return response_data

#########LSTM
##open model LSTM
file = open('C:/PENDIDIKAN/BINAR ACADEMY_DATA SCIENCE/Platinum/cl_platinum/resources_of_lstm/x_pad_sequences_lstm.pickle', 'rb')
feature_file_from_lstm = pickle.load(file)
file.close()

model_file_from_lstm = load_model('C:/PENDIDIKAN/BINAR ACADEMY_DATA SCIENCE/Platinum/cl_platinum/ModelLSTM/modelLSTM.h5')

##LSTM input teks
@swag_from("docs/LSTMForm.yml", methods=['POST'])
@app.route('/LSTMForm', methods=['POST'])

def LSTM():

    original_text = request.form.get('text')
    text1 = remove_emoji(original_text)
    text2 = remove_url(text1)
    text3 = clean_text(text2)
    text4 = clean_all(text3)
    text = word_best(text4)
    
    tokenizer.fit_on_texts(text)
    feature = tokenizer.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen = feature_file_from_lstm.shape[1])
    prediction = model_file_from_lstm.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]

    json_response = {
        'status_code': 200,
        'description': "LSTM Result of Sentiment Analysis",
        'data': {
            'text': original_text,
            'sentiment': get_sentiment
        },
    }
    response_data = jsonify(json_response)
    return response_data

##LSTM upload as file
@swag_from("docs/LSTMFile.yml", methods=['POST'])
@app.route('/LSTMFile', methods=['POST'])

def LSTM_FILE():

    fileok =  request.files.getlist('file')[0]
    df1 = pd.read_csv(fileok, names = ['text'], header = None)
    texts1 = df1.text.to_list()

    text_with_sentiment = []
    for original_text in texts1:
      text1 = remove_emoji(original_text)
      text2 = remove_url(text1)
      text3 = clean_text(text2)
      text4 = clean_all(text3)
      text_clean = word_best(text4)

      
      #text_feature = tfidf_vec.transform(text_clean)
      #get_sentiment = model_file_from_lstm.predict(text_feature)[0]

      text_with_sentiment.append(text_clean)
      
      tokenizer.fit_on_texts(text_clean)
      feature = tokenizer.texts_to_sequences(text_clean)
      feature = pad_sequences(feature, maxlen = feature_file_from_lstm.shape[1])
      prediction = model_file_from_lstm.predict(feature)
      get_sentiment = sentiment[np.argmax(prediction[0])]

      text_with_sentiment.append({
        'text':original_text,
        'sentiment': get_sentiment
      })

    json_response = {
      'status_code': 200,
      'description': "LSTM Result of Sentiment Analysis",
      'data': text_with_sentiment,
    }
    response_data = jsonify(json_response)
    return response_data


#error handler
@app.errorhandler(400)
def handle_400_error(_error):
  "the process is halted, the fault is from you"
  return make_response(jsonify({'error':'Misunderstood'}), 400)

@app.errorhandler(401)
def handle_401_error(_error):
   "the process is halted, it is unauthorized"
   return make_response(jsonify({'error':'Unauthorised'}), 401)

@app.errorhandler(404)
def handle_404_error(_error):
   "the process is halted, there is no such url"
   return make_response(jsonify({'error':'Not Found'}), 404)

@app.errorhandler(500)
def handle_500_error(_error):
   "the process is halted, server is occupied at the moment"
   return make_response(jsonify({'error':'Server error'}), 500)  


if __name__ == '__main__':
    app.run(debug = True) #Run & Show Error (if any)