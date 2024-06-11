# Speech Emotion Recognition

Speech emotion Analsis using LSTM, Transformer, implemented in Keras.
&nbsp;

## Environments

Recommend use virtual environment 
- Python 3.8
- Keras & TensorFlow 2


(1) Create the environment
```
python3.8 -m venv Speech-emotion-analysis 
```
(2)Activate
```
source lSpeech-emotion-analysis/bin/activate
```
(3)Check #Display the destination 
```
which python 
```





&nbsp;

## Structure

```
├── transformer_model.py    // transformer
├── transformer_train.py    // train transformer
├── transformer_predict.py  // prediction using transformer model
├── data_classify.py        // data processing
├── base.py                 // base model for transformer
├── lstm.py                 // LSTM
├── libro.py                // extract features using librosa
├── configs.py              // configure hyper parameters
├── datasets/               // store dataset
├── features/               // store extracted features
├── checkpoints/            // store model weights
├── train.py                // train
├── predict.py              // recognize the emotion of a given audio
└── preprocess.py           // data preprocessing (extract features and store them locally)
```

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

Every configuration is stored in [`configs.py`](https://github.com/Stoneyew/Speech-Emotion-Analysis/blob/main/configs.py).

&nbsp;

### Preprocess
Before extracting, you should classify the source_date to datasets.(source_file="source" to destination_directory="datasets")
```python
python data_classify.py
```

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

For example, if you want to predict a LSTM model:

```python
python predict.py
```

```python
Input audio_path:
```

Else, for transformers:

```python
python transformer_predict.py
```

&nbsp;

### TODO

Fix transformer
