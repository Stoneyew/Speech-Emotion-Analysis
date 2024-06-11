# config.py

# data
model = "lstm"

# dataset path 
data_path = "datasets"  
class_labels = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]  
# class_labels:
#   - "01"  # Neutral
#   - "02"  # Calm
#   - "03"  # Happy
#   - "04"  # Sad
#   - "05"  # Angry
#   - "06"  # Fearful
#   - "07"  # Disgust
#   - "08"  # Surprised
nums_labels = 8

# feature path
feature_folder = "features/8-category" 
feature_method = "l"  #use librosa


# checkpoint path
checkpoint_path = "checkpoints/"  
checkpoint_name = "check_point_lstm"  

# train configs
epochs = 20  # number of epoch
batch_size = 32  # batch size
lr = 0.001  # learn rate

# model set
rnn_size = 128  # LSTM hidden layer size
hidden_size = 32
dropout = 0.5
