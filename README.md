## Acoustic Machine Learning

This is the code base for running a very basic model on acoustic data based on my talk at Women in Data Science in Puget Sound, 2022. You can view the "Demystifying Acoustic Data for Dolphin Identification" slides [here](https://drive.google.com/file/d/1XU1lawZxnyl5oBEGfccrx5jPYj5ratUE/view?usp=sharing). It consists of data either in a shorter duration (a few seconds), or long audio recordings that come with a metadata file for segmentation.

I did not add code for data augmentation here because it is usually specific to the domain and needs to be discussed with the domain experts. Note that with acoustic data and spectrograms, we can not simply augment files as if they were images (i.e., you shouldn't flip a spectrogram upside down :)!)

With the use of CNN, the spectrograms are fixed in input image shape, however, it is beneficial to talk to scientists to determine the recommended ratio for kHz to inch and second to inch to determine figure sizes)

No code is perfect, so when you find a bug (a feature!), please let me know! If you can help make this README easier to follow, PLEASE let me know as well!

### Environment set up

1. Clone or download this repository to your local computer.
2. Unzip and/or change directory into the project folder.
3. This setup is tested with python 3.9 on a MacBook Pro running macOS Catalina. Create a new environment with `python -m virtualenv <env_name>`, `source <env_names>/bin/activate`, and `pip install -r <path_to_requirements.txt>`.


### Running the code
Adjust parameters in `generate_config.py`, specifically, specify `train` or `test` mode at the bottom under
```
# Mode for training/validation/testing/evaluation
    'mode': 'train',  # 'train' or 'test' (evaluate)
    'evaluate_after_training': True,  # to plot confusion matrix and loss plot
```
Then, in the root path
```
$ python generate_config.py
$ python main.py
```

Note: If you don't want the code to regenerate spectrograms every time it runs training, set create_spectrogrom to False in the generate_config.py file: `'create_spectrogram': False`


### Orcas or Dolphins?
I started this project with code I created previously on auditory data captured from patients with Parkinson's Disease. However those data are private and may not be shared externally. To show a mini data pipeline, I started with data from Orcasound Networks located in AWS S3. I have always wanted to volunteer my time with them so this was a nice way to familiarize myself with their work. My last work project was actually on dolphin identification, and our collaborators from Syracuse and WHOI have been most kind to support me in using their data for this talk. While I can't share the data here, they are considering open sourcing a labelled dataset to share with the community. I will update any links here if it becomes available. 

To run Orcasound data, check out their website and github repo (see presentation slides for the repo I've used) for the latest S3 location to obtain training and test data. You will want to update the input argument in main.py from "dolphins" to "orcas", and edit aws_s3.py to input the S3 bucket name and url. Once those are in place, running main.py will allow you to download data from the bucket.

Steps:
1. In main.py, update the input parameters from 

```main(download_from_s3=False, extract_file=False, dataset='dolphins')```

to 

`main(download_from_s3=True, extract_file=True, dataset='orcas')`

I recommend running `download_from_s3` and `extract_file` as a two-step process. This way you know which tar_file to untar and enter the file name in step 2 for `tar_file`.

2. In generate_config.py, update the parameters to enable extra new data and specify the tar file you want to untar. 
```
'data': {
    # Parameters for extracting new data from long audio files"
    'extract_new_data': True,  # Set this to true if you have metadata file to segment long audio files
    'tar_file': None,  # Set this to filename if you need to untar a tar.gz file
},
```
and 
```commandline
's3': {
    # All things Amazon s3
    'bucket_name': 's3_bucket',  # Your S3 bucket name
    'url': 's3_bucket_url'  # Your S3 URL where you data is located
}
```
3. Set the num_classes parameter in the generate_config.py file.
```'num_classes': 1,  # number of classes to classify```


4. Finally, set the class mode to binary, `'class_mode': 'categorical',  # 'binary' (see TF documentation for ImageDataGenerator`




### Directory structure
```
acoustic_ml
|-- README (this file!)
|-- generate_config.py (for generating config.json)
|-- main.py (main entry point of the codebase)
|-- acoustic_ml/
    |-- audio.py
    |-- aws_s3.py
    |-- data_loader.py (for WHOI's dolphin data)
    |-- data_loader_binary.py (for Orcasound's data)
    |-- data_parser.py (for segmenting long audio files)
    |-- evaluate.py (functions that will evaluate the model performance)
    |-- model_fn.py (where different models live!)
    |-- spectrogram.py (for creating spectrogram, mel-spectrogram, and pcen)
    |-- train.py (functions that will compile and train models)
    `-- visualization.py (plotting the confusion matrix, loss plots)
|-- tests/
    |-- test_data/ (directory where test data lives)
    |-- out/ (auto generated when running pytest)
    |-- test_audio.py
    |-- test_aws_s3.py (need to enter your own bucket and s3_url)
    |-- test_data_parser_binary.py
    `-- test_spectrogram.py
|-- data/
    |-- wav/
        |-- class1/
        |   |-- beautiful_soundfile.wav (or .flac)
        |   |-- more_beautiful_soundfile.wav
        |-- class2/
        |   |-- funny_music.wav
        |   `-- even_funnier_music.wav
        `-- class3/
```
