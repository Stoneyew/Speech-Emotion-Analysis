import os
import numpy as np
import libro as lf
from transformer_model import TransformerSER
import plot
import configs
import warnings
warnings.filterwarnings("ignore")

def predict(config, audio_path: str, model) -> None:
    """Predict the emotion from an audio file."""

    # Extract features from the audio file
    test_feature = lf.get_data(config, audio_path, train=False)

    # Predict emotion and probabilities
    result = model.predict(test_feature)
    result_prob = model.predict_proba(test_feature)

    print('Recognition: ', config.class_labels[int(result)])
    print('Probability: ', result_prob)
    
    # Plot the probabilities
    plot.radar(result_prob, config.class_labels)

if __name__ == '__main__':
    audio_path = '/Users/angus/Downloads/新錄音 22.m4a'
    
    # Load configuration
    # audio_path = input("Input audio_path: ")
    # if os.path.exists(audio_path):
    #     print("文件存在")
    # else:
    #     print("文件不存在")
    
    # Load the transformer model
    model = TransformerSER.load(configs.checkpoint_path, configs.checkpoint_name)
    
    # Perform prediction
    predict(configs, audio_path, model)
