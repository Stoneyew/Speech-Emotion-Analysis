import os
import numpy as np
import libro as lf
from lstm import DNN, LSTM
import matplotlib.pyplot as plt
import configs
import warnings
warnings.filterwarnings("ignore")

def plot_radar(data_prob: np.ndarray, class_labels: list) -> None:

    angles = np.linspace(0, 2 * np.pi, len(class_labels), endpoint=False)

    
    data = np.concatenate((data_prob, [data_prob[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    class_labels = class_labels + [class_labels[0]]

    fig = plt.figure()
    
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, data, "bo-", linewidth=2)
    ax.fill(angles, data, facecolor="r", alpha=0.25)
    ax.set_thetagrids(angles * 180 / np.pi, class_labels)
    ax.set_title("Emotion Recognition", va="bottom")

    
    ax.set_rlim(0, 1)

    ax.grid(True)
    # plt.ion()
    plt.show()
    # plt.pause(4)
    # plt.close()


def convert_to_percentages(numbers):
    return [f'{num * 100:.2f}%' for num in numbers]

def predict(config, audio_path: str, model) -> None:
    """ predict the emotion frequency """

    # utils.play_audio(audio_path)


    test_feature = lf.get_data(config, audio_path, train=False)

    result = model.predict(test_feature)
    result_prob = model.predict_proba(test_feature)
    print('Recogntion: ', configs.class_labels[int(result)])
    
    percentages = convert_to_percentages(result_prob)
    max_class_length = max(len(label) for label in configs.class_labels)
    max_prob_length = max(len(prob) for prob in percentages)
    print('\n')
    print(f"{'class':<{max_class_length}}   {'probability':<{max_prob_length}}")
    print('-' * (max_class_length + max_prob_length + 3))
    for label, percentage in zip(configs.class_labels, percentages):
        print(f"{label:<{max_class_length}}   {percentage:<{max_prob_length}}")
    plot_radar(result_prob, config.class_labels)


if __name__ == '__main__':
    # audio_path = '/Users/angus/Downloads/新錄音 22.m4a'
    

    while True:
        audio_path = input("Input audio_path: ")
        if os.path.exists(audio_path):
            print("EXIST! START PREDICT")
            print("-----------------------------------------------------------------")
            break  
        else:
            print("Not exist. Plz input again")

    model = LSTM.load(configs.checkpoint_path,configs.checkpoint_name)
    predict(configs, audio_path, model)
