class HParams:
    # Tacotron2 Hyperparameters
    tacotron_num_mels = 80
    tacotron_num_freq = 1025
    tacotron_sample_rate = 22050
    tacotron_frame_length_ms = 50
    tacotron_frame_shift_ms = 12.5
    tacotron_num_mgcs = 60
    tacotron_preemphasis = 0.97
    tacotron_min_level_db = -100
    tacotron_ref_level_db = 20
    epochs = 3  # Number of epochs for training
    batch_size = 8  # Reduced batch size for memory efficiency
    learning_rate = 0.001  # Learning rate for training

    # WaveNet Hyperparameters
    wavenet_layers = 20
    wavenet_blocks = 2
    wavenet_residual_channels = 64
    wavenet_dilation_channels = 64
    wavenet_skip_channels = 256
    wavenet_kernel_size = 2
