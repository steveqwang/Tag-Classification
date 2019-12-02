# StackOverflowTag-Classcification
This is a text-Classification task to classify the tags of [azure, aws, gcp], based on the title of StackOverflow questions.  

Data is scraped from StackOverflow.  

## Dataset Details
* Azure: 77216 titles
* AWS: 82182 titles
* GCP: 17576 titles

## Requirements:
* Python 3.7
* TensorFlow 2.0
* Scikit-Learn
## Machine-Learning Models

## Deep-Learning Models
Several Deep Models are implemented  
### Training Details
* steps: 3 epochs
* batch_size: 64
* embedding_size: 128
* max_seq_length: 20
* RNN_dimension: 128
* dropout_rate: 0.5
* testing_size: 0.15
### Models
#### FastText
* Accuracy on training: 0.9506
* Accuracy on testing: 0.9442
#### TextCNN
* Accuracy on training: 0.9543
* Accuracy on testing: 0.9473
#### TextRNN
* Accuracy on training: 0.9521
* Accuracy on testing: 0.9462
#### TextRCNN
* Accuracy on training: 0.9626
* Accuracy on testing: 0.9478
#### SelfAttention
* Accuracy on training: 0.9541
* Accuracy on testing: 0.9453