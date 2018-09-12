# Music Speech Classifier

## Abstract

Aim is to implement a classifier which classifies an audio sample into speech
or music. The dataset which will be used for the project will be theGTZAN
Music/Speech collection and theColumbiaMusic/Speech dataset.Marsyas v0.
library will be used to identify and extract relevant features from the audio samples.
A number of methods are implemented in an attempt to find the best performing
classifier. Performance is compared for Naive Bayes, Decision Tree, K-Nearest
neighbor, SVM, 3-layer Neural network, an Ensemble of all previously listed
classifiers, and finally a Convolutional Neural Network. PCA is performed to
minimize the number of features. Accuracy of each classifier is examined before
and after performing PCA.

## 1 Introduction

Music-Speech discrimination is a popular problem in the multi-media domain. One most common
example is to apply different types of compression to speech and music in a stream to compress the
final audio stream more efficiently.
Feature extraction from audio streams is one of the more difficult parts of the problem since a variety
of different features can be extracted based on requirement. The different features which can be
extracted are listed below -

- Spectral Centroid: This feature is associated with musical timbre and is calculated by taking
    the weighted average of the power spectrum with respect to the frequencies.
- Spectral Flux : This feature indicates the rate at which the power spectrum of a signal is
    changing, by taking differences between the adjacent values in the spectrum.
- Zero Crossing Rate : As its name suggests, the zero crossing rate is the rate at which a given
    signal changes sign.
- MFCCs : Mel Frequency Cepstrum Coefficients (MFCC) are one of the widely used features
    in Music Information Retrieval. They are commonly derived by first taking the fourier
    transform, mapping it to the mel scale and then taking DCT of the mel log powers.
- Chroma Features : These features encode the short-time energy distribution of music signal
    over 12 traditional pitch classes of equal-tempered scale.

## 2 Methodology

Marsyas(Music Analysis, Retrieval, and Synthesis for Audio Signals) is an open source software
framework for audio processing with specific emphasis on Music Information Retrieval Applications.
It is an excellent framework for rapid prototyping and experimentation with audio files. This library
is used for feature extraction from the GTZAN and Columbia Music/Speech collections.


scikit-learnis a simple and efficient tool for data analysis and machine learing. It is built on top
of popular python libraries like Numpy, SciPy, and matplotlib. All classifiers except the CNN, are
implemented using scikit in this project.

Kerasis one the popular deep learning libraries running on top of TensorFlow, CNTK, or Theano. It
was developed with a focus on enabling fast experimentation and allows for easy and fast prototyping.
It was used to implement the CNN classifier in this project.

2.1 Dataset

The dataset is a combination of two different collections: GTZAN[5] and the Columbia dataset[6].

- GTZAN: GTZAN contains 128 audio clips evenly distributed between music and speech,
    each 30 seconds in duration.
- Columbia: Contains 80 speech samples and 81 music samples each of 15 seconds in duration.

Since there is a mismatch in the duration of clips in both datasets, the GTZAN samples are trimmed
to 15 seconds to match samples from the Columbia dataset.
After going through existing literature on the subject [1][2], MFCCs were decided to be the feature of
choice for the aim of this project. The Marsyas library was used to extract this feature from the entire
combined dataset and stored in an.arfffile further processing. This file also contains the class labels
of music/speech.
The python scriptmusic_speech_feature_extractor.pyis used to extract individual samples
from this.arfffile and stores it in the form of.npynumpy array. This .npy file is used as input for all
classifiers below. For explanation on how the features were extracted from audio samples, please
referReadme.txtin code directory.
By the process of experimentation it was found that standardizing the dataset only worsened the
accuracy of the classifiers. Hence, the extracted features were used as is without any normalization.

2.2 Classifiers

As mentioned above, scikit and Keras python libraries were used for building the classifiers. Accuracy
of each classifier is measured using 4-fold cross validation on the entire dataset. Principal Component
Analysis(PCA) reduces the feature space from 60 features to 11 features. The impact on accuracy
before and after PCA can be visualized in the Results section. Implementation details for each of the
classifiers are as below -

2.2.1 Naive Bayes

A simple Naive Bayes classifier. Accuracy is found to be92.37%. The confusion matrix is as below -

```
()music speech
music 136 9
speech 13 131
```
The accuracy after PCA is93.06% and is the only case where accuracy improved after PCA.

2.2.2 Decision Tree

A simple Decision tree classifier. Accuracy is found to be 91 %. The confusion matrix is as below -

```
()music speech
music 136 9
speech 11 133
```
The accuracy after PCA is91.67% and stays roughly the same after performing PCA.


2.2.3 K-Nearest Neighbor

The value ofkwhich performs the best for this dataset is 6. Accuracy is found to be 95.48%. The
confusion matrix is as below -

```
()music speech
music 137 8
speech 5 139
```
The accuracy after PCA is94.79%.

### 2.2.4 SVM

A soft margin SVM classifier withC= 0. 1. Linear kernal was used for simplicity. Accuracy is found
to be96.53%. The confusion matrix is as below -

```
music speech
()music 140 5
speech 5 139
```
The accuracy after PCA is95.84%.

2.2.5 Simple Neural Network

A 4 layer neural network with 1 input layer, 2 hidden layers, and 1 output layer. RELU is the
activation function withαas 10 −^5. Accuracy is found to be94.10%. The confusion matrix is as
below -

```
music speech
()music 134 11
speech 6 138
```
The accuracy after PCA is92.71%.

2.2.6 Ensemble

The classifier was a combination of Naive Bayes, SVM, KNN, decision tree, and the Simple neural
network classifiers. A voting strategy was implemented. The final output is decided based on the
majority results of each of individual classifiers. Accuracy is found to be96.18%. The confusion
matrix is as below -

```
music speech
()music 140 5
speech 5 139
```
The accuracy after PCA is96.53%.

2.2.7 Convolutional Neural Network

Accuracy is found to be98.85%. The samples where divided into training & test batches of 202 & 87
respectively. The model was trained for 15 epochs and mini-batch size was set to be 50. Categorical
cross-entropy was used as the loss function, while the Adam optimizer was used for optimization.


## 3 Results

From above to accuracies, it is obvious that there are lot of redundant features in the dataset and
accuracy only drops a little with huge drop in feature space.

```
Figure 1: Accuracies of all classifiers without PCA, i.e., feature space = 60
```
From the above figure, it is clear that CNN is the best performing classifier with accuracy of98.85%.
SVM takes the second place with96.53% accuracy.

```
Figure 2: Accuracies of all classifiers except CNN with PCA, i.e., feature space = 11
```
The CNN classifier ran into difficulties after performing PCA. Hence, it is not considered in the above
graph. The best performing classifier for the reduced feature space were the Ensemble classifier and
though SVM classifiers.

## 4 Conclusion

The effect of PCA on CNN needs to be implemented and verified. The SVM classifier provides the
most consistent performance among all classifiers for this particular dataset. CNN gives the best
accuracy but also takes the longest time in training stage.


## 5 References

[1] G. Tzanetakis and P. Cook, "Musical genre classification of audio signals," inIEEE Transactions on Speech
and Audio Processing, vol. 10, no. 5, pp. 293-302, Jul 2002.doi: 10.1109/TSA.2002.

[2] Logan, Beth. ”Mel Frequency Cepstral Coefficients for Music Modeling.” ISMIR. 2000.

[3] Kim, Kibeom, et al., Speech Music Discrimination Using an Ensemble of Biased Classifiers,Audio Engi-
neering Society Convention 139. Audio Engineering Society, 2015.

[4] Mandel, M., Ellis, D.. ”Song-Level Features and SVMs for Music Classification”
[http://www.ee.columbia.edu/~dpwe/pubs/ismir05-svm.pdf.](http://www.ee.columbia.edu/~dpwe/pubs/ismir05-svm.pdf.)

[5] George Tzanetakis, GTZAN Music/Speech Collection, University of Victoria

[6] Dan Ellis, The Music-Speech Corpus, Columbia University, 2006.