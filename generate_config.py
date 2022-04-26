"""This scripts allow us to generate configuration file"""
import json


def generate_config(config_filename='config.json'):
    config = {
        'logging': {
            # Logging variables - not fully utilized, been using print statements
            'file_name': 'main.log',
            'level': 'DEBUG'
        },
        'paths': {
            # Paths and metadata
            'data_path': 'data',  # root_path of the data
            'wav_path': 'wav',  # location of the wav files
            'output_path': 'outputs',  # this is where spectrogram, checkpoints, and saved model will go
            'pos_spec_path': 'spectrogram/pos_spec',  # code will add this to the output_path
            'neg_spec_path': 'spectrogram/neg_spec',  # code will add this to the output_path
            'checkpoint_path': 'checkpoints_dolphin',  # path where checkpoints and the model are saved
            'metadata': 'train.tsv'  # code will add this to the base_path where file is the tar file is extracted to
        },
        's3': {
            # All things Amazon s3
            'bucket_name': 's3_bucket',  # Your S3 bucket name
            'url': 's3_bucket_url'  # Your S3 URL where you data is located
        },
        'gcs': {
            # placeholder for accessing gcs data via APIs (Sanctsound data is stored in GCS)
        },
        'data': {
            # Parameters for extracting new data from long audio files"
            'extract_new_data': False,  # Set this to true if you have metadata file to segment long audio files
            'tar_file': None,  # Set this to filename if you need to untar a tar.gz file
        },
        'audio': {
            # Parameters for processing audio files
            'sampling_rate': 60000,  # set to None for reading in native sampling rate
            'duration': 2  # read in the audio file for this many seconds as a single segment
        },
        'spectrogram': {
            # Parameters for creating spectrogram (keep this simple and basic for demonstration)
            'create_spectrogram': True,  # set True to create spectrograms from audio data
            'nfft': 1024,  # number of Fourier Transform bins
            'window': 'hamming',  # windowing function prior to taking the Fourier Transform
            'max_length_sec': 2,  # pad the time dimension of the spectrogram to this
            'percentile_threshold': 50,  # post processing to improve dynamic range
            'image_shape': None  # Leave this as None, will automatically populate when creating the spectrogram
        },
        'melspec': {
            # Parameters for creating mel-spectrogram (more appropriate for low frequency signals)
            'fmin': 0,  # min frequency in Hz
            'fmax': 10000,  # max frequency in Hz, if None, it will use sampling rate / 2
            'n_mels': 128,  # number of mel bands to generate
        },
        'pcen': {
            # Parameters for per-channel-energy-normalization (awesome for detection, i.e triplet loss)
            'gain': 0.8,  # gain factor <1
            'bias': 8,  # bias point of the nonlinear compression
            'power': 0.5,  # compression constant between 0 and 1
            'time_constant': 0.05,  # time constant for IIR filter
        },
        'model': {
            # Parameters for models
            'type': 'built-in',  # 'built-in' or a string != 'built-in'
            'num_classes': 4,  # number of classes to classify
            'epochs': 500,  # maximum number of epochs to train on
            'batch_size': 16,  # batch size for training and backprop
            'learning_rate': 1e-4,  # learning rate (using optimizer's default decay parameters)
            'patience': 10,  # for early stopping
            'num_freeze_layers': 100,  # we can freeze number of layers to reduce number of trainable parameters
            'class_mode': 'categorical',  # 'binary' or 'categorical' (see TF documentation for ImageDataGenerator)
        },
        # Mode for training/validation/testing/evaluation
        'mode': 'train',  # 'train' or 'test' (evaluate)
        'evaluate_after_training': True,  # if True, plot confusion matrix and loss plot
        'test_path': 'outputs/spectrogram/val'  # set this if mode is test
    }

    with open(config_filename, 'w') as f:
        json.dump(config, f, indent=4)


if __name__ == "__main__":
    generate_config()
