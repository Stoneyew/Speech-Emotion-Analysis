import matplotlib.pyplot as plt

def plot_mfcc(mfcc):
    plt.figure(figsize=(10, 4))
    plt.imshow(mfcc, aspect='auto', origin='lower')
    plt.title('MFCC')
    plt.colorbar()
    plt.tight_layout()
    plt.show()
