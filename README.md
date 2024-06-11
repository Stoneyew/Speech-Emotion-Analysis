# Speech Emotion Recognition

Speech emotion recognition using LSTM, Transformer, implemented in Keras.

We have improved the feature extracting method and achieved higher accuracy (about 80%).

&nbsp;

## Environments

- Python 3.8
- Keras & TensorFlow 2

&nbsp;

## Structure

```
├── models/                // models
│   ├── common.py          // base class for all models
│   ├── dnn                // neural networks
│   │   ├── dnn.py         // base class for all neural networks models
│   │   ├── cnn.py         // CNN
│   │   └── lstm.py        // LSTM
│   └── ml.py              // SVM & MLP
├── extract_feats/         // features extraction
│   └── libro.py         // extract features using librosa 
├── utils/
│   ├── files.py           // setup dataset (classify and rename)
│   ├── opts.py            // argparse
│   └── plot.py            // plot graphs
├── configs/               // configure hyper parameters
├── features/              // store extracted features
├── checkpoints/           // store model weights
├── train.py               // train
├── predict.py             // recognize the emotion of a given audio
└── preprocess.py          // data preprocessing (extract features and store them locally)
```

&nbsp;

## Requirments

### Python

&nbsp;

## Datasets

1. [RAVDESS](https://zenodo.org/record/1188976)

   English, around 1500 audios from 24 people (12 male and 12 female) including 8 different emotions (the third number of the file name represents the emotional type): 01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised.

&nbsp;

## Usage

### Prepare

Install dependencies:

```python
pip install -r requirements.txt
```

&nbsp;

### Configuration

Every configuration is stored in [`configs.py`](configs).

&nbsp;

### Preprocess

First of all, you should extract features of each audio in dataset and store them locally. Features extracted by librosa will be saved in `.p` files.

```python
python preprocess.py
```

&nbsp;

### Train

Audios which express the same emotion should be put in the same folder (you may want to refer to [`data_classify.py`] when setting up datasets), for example:

```
└── datasets
    ├── angry
    ├── happy
    ├── sad
    ...
```

For example, if you want to train a LSTM model:

```python
python train.py
```

Else, for transformers:

```python
python transformer_train.py
```

&nbsp;

### Predict

This is for when you have trained a model and want to predict the emotion for an audio. Check out [`checkpoints/`](https://github.com/Anguschen0430/AI_final_project/tree/main/checkpoints) for some checkpoints.

First modify following things in [`predict.py`](predict.py):

```python
audio_path = 'str: path_to_your_audio'
```

For example, if you want to predict a LSTM model:

```python
python predict.py
```

Else, for transformers:

```python
python transformer_predict.py
```

&nbsp;

### Functions

#### Radar Chart

Plot a radar chart for demonstrating predicted probabilities.

Source: [Radar](https://github.com/Zhaofan-Su/SpeechEmotionRecognition/blob/master/leidatu.py)

```python
import plot

"""
Args:
    data_prob (np.ndarray): probabilities
    class_labels (list): labels
"""
plot.radar(data_prob, class_labels)
```

&nbsp;

#### Play Audio

```python
import plot

plot.play_audio(file_path)
```

&nbsp;

#### Plot Curve

Plot loss curve or accuracy curve.

```python
import plot

"""
Args:
    train (list): loss or accuracy on train set
    val (list): loss or accuracy on validation set
    title (str): title of figure
    y_label (str): label of y axis
"""
plot.curve(train, val, title, y_label)
```

&nbsp;

#### Waveform

Plot a waveform for an audio file.

```python
import plot

plot.waveform(file_path)
```

&nbsp;

#### Spectrogram

Plot a spectrogram for an audio file.

```python
import plot

plot.spectrogram(file_path)
```
