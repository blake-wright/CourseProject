# Classification Competition - Analyzing Twitter Tweets

Video link to presentation: 

https://drive.google.com/file/d/17dMSY3kKD93lZayn_cvysVckzSeQaHD7/view?usp=sharing

In the video I go over most of this README besides the setup. I do have another video below that is my own video of the environment setup.



Video link to environment setup (farther down you will see the videos from Jeff Heaton where I sourced this information): 

https://drive.google.com/file/d/1hQFQuth2hXUuuUFBRt_a4RNttmnmoQ8x/view?usp=sharing

conda env create -v -f tensorflow.yml

python -m ipykernel install --user --name tensorflow --display-name "Python 3.7 (tensorflow)"

## Setting up your environment

You will need the following libraries to successfully run my project:

Library        Version Used    Pip install cmd
--------------------------------------------------------
Tensorflow     2.3.1           pip install tensorflow
Sklearn-learn  0.23            pip install sklearn
Transformers   3.5.1           pip install transformers
Pandas         1.1.3           pip install pandas
Numpy          1.18.5          pip install numpy
Torch          1.7.1           pip install torch


The following videos can be used as a reference on how to setup a miniconda python environment if you
don't have any of the libraries and want some setup automatically. However, for the tensorflow.yml file
you will want to update the ```tensorflow=2.0``` to ```tensorflow=2.3.1```. Or just copy the below.

```
name: tensorflow

dependencies:
    - python=3.7
    - pip>=19.0
    - jupyter
    - tensorflow=2.3.1
    - scikit-learn
    - scipy
    - pandas
    - pandas-datareader
    - matplotlib
    - pillow
    - tqdm
    - requests
    - h5py
    - pyyaml
    - flask
    - boto3
    - pip:
        - bayesian-optimization
        - gym
        - kaggle
```

For Windows:
https://www.youtube.com/watch?v=RgO8BBNGB8w

For MacOS:
https://www.youtube.com/watch?v=MpUvdLD932c&t=372s

## Running the project
Before trying to run the project note that this is a very resource demanding program. I have tested it on the following pieces of hardware.

* Desktop:
    * CPU: 3.9 GHz Ryzen 7 3800X
    * RAM: 32GB
    * DISK: < 10GB available
* Laptop:
    * CPU: 3.1 GHz i5
    * RAM: 16GB
    * DISK: < 200GB available

If your hardware isn't able to run please contact me and you can use my system. I am working on additional ways to test on lower RAM devices.

To run this project:
If you have jupyter notebook you can locate the file, select file, and choose the ```Cell``` tab and then select ```Run All```.

## Structure of the project

I first imported the data that was given by using the pandas library.
```
#read in train & test data
trainData = pd.read_json('train.jsonl', lines = True)
testData = pd.read_json('test.jsonl', lines = True)
```

Next I converted the labels ("SARCASM" and "NOT_SARCASM") to binary values. This was done because the model requires a binary label to respond to.
```
for i in range(len(trainData)):
    if trainData['label'][i] == "SARCASM":
        trainData['label'][i] = 1
    else:
        trainData['label'][i] = 0
```

From here I set my modelId, tokenizer, and model. 

I choose to include case and I felt that sometimes when people are sending out sarcastic tweets they may often use letter case to
further voice their sarcasm.

As you can see I used the AutoTokenizer from the transformers library as I would not have to switch it when using different models.

I initiailly wrote this project using DistilBert, which is far faster than Bert and is almost as accurate.
I also did try using BERT, I did get better results (~2% accuracy) but when I uploaded them to the leaderboard they were slightly worse. I also tried XLNet and RoBERTa but with my code they were performing up to 10% less accurate than BERT. I believe this was because I was not able to configure them as precisely.

```
#setting modelId, tokenizer, and model
modelId = "distilbert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(modelId)
model = DistilBertModel.from_pretrained(modelId)
```

Next up was getting all of the data ready for the model. Tokenizing the data, padding the lengths, and masking so the padding was not used.
```
tokenized = trainData['response'].apply((lambda x: tokenizer.encode(x, add_special_tokens = True)))
tokenized_2 = testData['response'].apply((lambda x: tokenizer.encode(x, add_special_tokens = True)))

trainPad = pad_sequences(tokenized, maxlen = 100, padding='post')
testPad = pad_sequences(tokenized_2, maxlen = 100, padding='post')

trainMask = np.where(trainPad != 0,1,0)
testMask = np.where(testPad != 0,1,0)

#converting to int64
trainInput = torch.tensor(trainPad).to(torch.int64)
testInput = torch.tensor(testPad).to(torch.int64)

#converting to tensor type
trainMask = torch.tensor(trainMask)
testMask = torch.tensor(testMask)
```

I concurrently modeled both the training set and the testing set. I have done these concurrently because upon prediction the
input will have to match what it was trained against.

```
with torch.no_grad():
    output = model(trainInput, attention_mask = trainMask)
with torch.no_grad():
    outputTest = model(testInput, attention_mask = testMask)
```

Here I prepared features and labels that will be used to train and test.
```
trainFeats = output[0][:,0,:].numpy()
testFeats = outputTest[0][:,0,:].numpy()
labels = trainData['label']
trainFeats, valFeats, train_labels, test_labels = train_test_split(trainFeats, labels)
```

I tried many, many different classifiers and RandomForestClassifier teneded to fair the best at ~77%.
77% accuracy was lower than expected and desired. I will expand on improvements in the future in the 'Improvement' section.

```
classifier = RandomForestClassifier(n_estimators=500, max_depth=None,
                                         min_samples_split=8, random_state=2)
classifier.fit(trainFeats, train_labels)
classifier.score(valFeats, test_labels)
```

For the rest of the program I am just creating the answer file and writing to it. There is a small bug where it sometimes does not write 
the entire file. I am troubleshooting this problem. I included a print screen so you could see the results being written.
```
results = classifier.predict(testFeats)

f = open("answer.txt", "w")
print(results[1799])
for i in range(len(results)):
    id = str(testData['id'][i])
    answer = str(results[i])
    f.write(id)
    f.write(',')
    if answer == "1":
        f.write("SARCASM")
    else:
        f.write("NOT_SARCASM")
    f.write('\n')
    print(id, answer)
```

## Improvements

I spent a large (maybe too much) time trying to train the model instead of using a pretrained model for BERT. I was not very successful in this and wish I
could have had more time to expand on it as I feel like this would have greatly increased results. 

I also considered manipulating the tweets. Some things I considered were taking out common words that would be used in a sarcastic or not sarcastic
and throwing them out as my model may have been able to better train on relevant information. There was also consideration on furthering expanding weighting
on the hashtags found in tweets as many of them seemed to correlate strongly from my point of view. However, I was not sure how to go about this.

I did not use the context as part of the analysis as well. I feel like this could have been a big improvement without too much more additional code.
However, time was a factor in this project and I was not able to complete this task.

## Credit/Documentation

Huggingface's website was a great help. The transformers library is maintained by them which was used in this project. They also provide ample of documentation
on how to use them. I also found their examples extremely useful in understanding the flow of the program. I have included links to both the home page and to the example.

https://huggingface.co/

https://huggingface.co/transformers/model_doc/distilbert.html (Documentation on library)

https://github.com/huggingface/notebooks/blob/master/examples/text_classification.ipynb

Jay Alammar was also a great help to this project. His visual guide helped fill in knowledge gaps of how each part of the model worked and which types it needed.
(http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)
