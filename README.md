# Next_Word_Prediction_Using_Neural_Networks
Next word prediction is a language modelling task in Machine Learning that aims to predict the most probable word or sequence of words that follows a given input context. This task utilizes statistical patterns and linguistic structures to generate accurate predictions based on the context provided.
Next Word Prediction means predicting the most likely word or phrase that will come next in a sentence or text. It is like having an inbuilt feature on an application that suggests the next word as you type or speak.
The Next Word Prediction Models are used in applications like messaging apps, search engines, virtual assistants, and autocorrect features on smartphones.
To build a Next Word Prediction model:
  Start by collecting a diverse dataset of text documents, 
  Preprocess the data by cleaning and tokenizing it, 
  Prepare the data by creating input-output pairs, 
  Engineer features such as word embeddings, 
  Select an appropriate model like an LSTM or GPT, 
  Train the model on the dataset while adjusting hyperparameters,
  Improve the model by experimenting with different techniques and architectures.

Some Code Snippet Explaination:

  tokenizer = Tokenizer()                               
  tokenizer.fit_on_texts([text])
  total_words = len(tokenizer.word_index) + 1
In the above code, the text is tokenized, which means it is divided into individual words or tokens. The ‘Tokenizer’ object is created, which will handle the tokenization process. The ‘fit_on_texts’ method of the tokenizer is called, passing the ‘text’ as input. This method analyzes the text and builds a vocabulary of unique words, assigning each word a numerical index. The ‘total_words’ variable is then assigned the value of the length of the word index plus one, representing the total number of distinct words in the text.

input_sequences = []
for line in text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)
In the above code, the text data is split into lines using the ‘\n’ character as a delimiter. For each line in the text, the ‘texts_to_sequences’ method of the tokenizer is used to convert the line into a sequence of numerical tokens based on the previously created vocabulary. The resulting token list is then iterated over using a for loop. For each iteration, a subsequence, or n-gram, of tokens is extracted, ranging from the beginning of the token list up to the current index ‘i’.
This n-gram sequence represents the input context, with the last token being the target or predicted word. This n-gram sequence is then appended to the ‘input_sequences’ list. This process is repeated for all lines in the text, generating multiple input-output sequences that will be used for training the next word prediction model. 

  max_sequence_len = max([len(seq) for seq in input_sequences])
  input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
In the above code, the input sequences are padded to ensure all sequences have the same length. The variable ‘max_sequence_len’ is assigned the maximum length among all the input sequences. The ‘pad_sequences’ function is used to pad or truncate the input sequences to match this maximum length.

The ‘pad_sequences’ function takes the input_sequences list, sets the maximum length to ‘max_sequence_len’, and specifies that the padding should be added at the beginning of each sequence using the ‘padding=pre’ argument. Finally, the input sequences are converted into a numpy array to facilitate further processing.

y = np.array(tf.keras.utils.to_categorical(y, num_classes=total_words))
In the above code, the output array is converted into a suitable format for training a model, where each target word is represented as a binary vector.

model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(LSTM(150))
model.add(Dense(total_words, activation='softmax'))
print(model.summary())
The code above defines the model architecture for the next word prediction model. The ‘Sequential’ model is created, which represents a linear stack of layers. The first layer added to the model is the ‘Embedding’ layer, which is responsible for converting the input sequences into dense vectors of fixed size. It takes three arguments:

‘total_words’, which represents the total number of distinct words in the vocabulary; 
‘100’, which denotes the dimensionality of the word embeddings; 
and ‘input_length’, which specifies the length of the input sequences.
The next layer added is the ‘LSTM’ layer, a type of recurrent neural network (RNN) layer designed for capturing sequential dependencies in the data. It has 150 units, which means it will learn 150 internal representations or memory cells.

Finally, the ‘Dense’ layer is added, which is a fully connected layer that produces the output predictions. It has ‘total_words’ units and uses the ‘softmax’ activation function to convert the predicted scores into probabilities, indicating the likelihood of each word being the next one in the sequence.

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=100, verbose=1)
In the above code, the model is being compiled and trained. The ‘compile’ method configures the model for training. The ‘loss’ parameter is set to ‘categorical_crossentropy’, a commonly used loss function for multi-class classification problems. The ‘optimizer’ parameter is set to ‘adam’, an optimization algorithm that adapts the learning rate during training.

The ‘metrics’ parameter is set to ‘accuracy’ to monitor the accuracy during training. After compiling the model, the ‘fit’ method is called to train the model on the input sequences ‘X’ and the corresponding output ‘y’. The ‘epochs’ parameter specifies the number of times the training process will iterate over the entire dataset. The ‘verbose’ parameter is set to ‘1’ to display the training process.

The above code will take more than an hour to execute.

seed_text = "I will leave if they"
next_words = 3

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = np.argmax(model.predict(token_list), axis=-1)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word

print(seed_text)
The above code generates the next word predictions based on a given seed text. The ‘seed_text’ variable holds the initial text. The ‘next_words’ variable determines the number of predictions to be generated. Inside the for loop, the ‘seed_text’ is converted into a sequence of tokens using the tokenizer. The token sequence is padded to match the maximum sequence length.

The model predicts the next word by calling the ‘predict’ method on the model with the padded token sequence. The predicted word is obtained by finding the word with the highest probability score using ‘np.argmax’. Then, the predicted word is appended to the ‘seed_text’, and the process is repeated for the desired number of ‘next_words’. Finally, the ‘seed_text’ is printed, which contains the initial text followed by the generated predictions.
  
