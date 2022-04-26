"""Main entry point of the program"""
import os
from pathlib import Path
import logging
import json
import random


from acoustic_ml import aws_s3
from acoustic_ml import data_loader_binary, data_parser_binary  # for orcas
from acoustic_ml import data_loader  # for dolphin
from acoustic_ml import train, evaluate

random.seed(42)


def main(download_from_s3: bool = False, extract_file: bool = False, dataset: str = 'orcas'):
    """Load configuration file and set up paths"""

    # ----------------------------------------------------------------------------------------------------
    # Load configuration file and set up path
    # ----------------------------------------------------------------------------------------------------
    with open('config.json', encoding="utf-8") as json_f:
        cfg = json.load(json_f)

    data_path = Path(cfg['paths']['data_path'])

    # ----------------------------------------------------------------------------------------------------
    # Set up logging
    # ----------------------------------------------------------------------------------------------------
    logging.basicConfig(filename=cfg['logging']['file_name'], filemode='w',
                        format='"%(name)s - %(levelname)s - %(message)s',
                        level=cfg['logging']['level'])
    logging.getLogger('numba').setLevel('WARNING')
    logging.getLogger('matplotlib').setLevel('WARNING')
    logger = logging.getLogger()
    logger.info("Logging set up")

    # ----------------------------------------------------------------------------------------------------
    # download data from s3
    # ----------------------------------------------------------------------------------------------------
    if download_from_s3:
        # Download from S3 (last tested on 25.04.2022)
        output_path = data_path
        s3 = aws_s3.AWSS3(cfg['s3']['bucket_name'], cfg['s3']['url'], output_path)
        print(f"Downloading from {s3.get_bucket_name()} bucket")
        s3.download_s3_files()

    # ----------------------------------------------------------------------------------------------------
    # Untar downloaded tar.gz file if needed
    # ----------------------------------------------------------------------------------------------------
    if extract_file:
        # Extract tar file (last checked 25.04.2022 and 15.03.2022)
        # Entered the tar.gz file name in the config file so we won't untar everything
        tar_path = aws_s3.untar_file(Path(os.path.join(data_path, cfg['data']['tar_file'])))
        cfg['paths']['metadata'] = tar_path / cfg['paths']['metadata']

    if dataset == 'orcas':
        # ----------------------------------------------------------------------------------------------------
        # Extract files and obtain positive and negative samples
        # ----------------------------------------------------------------------------------------------------
        # uncomment this if the data is of a long audio sequence with metadata for segmenting
        data_parser_binary.DataParserBinary(cfg)

        # ----------------------------------------------------------------------------------------------------
        # If data is already in a positive and negative folders (or class folders),
        # we can load or process the data from wav to jpg directly without calling DataParser which
        # segments long audio files
        # Folder directory
        # |--outputs
        #       |--positive
        #              |--positive1.jpg
        #              |--positive2.jpg
        #       |--negative
        #              |--negative1.jpg
        #              |--negative2.jpg
        # Uncomment the line below to create the above directory structure for data
        # ----------------------------------------------------------------------------------------------------
        data_loader_obj = data_loader_binary.DataLoaderBinary(cfg)
        data_loader_obj.create_spectrograms()

    if dataset == 'dolphins':
        data_loader_obj = data_loader.DataLoader(cfg)

        # Run this if we need to create spectrogram from wav files (or bypass to save time)
        # this could be done in GPU at run time as well!
        if cfg['spectrogram']['create_spectrogram']:
            data_loader_obj.create_spectrograms()

    # Check input shape if it is not yet defined
    # Normally defined when creating the spectrogram, unless we bypass the step
    # Perhaps move to helper functions (need to think more about how best to structure this)
    if cfg['spectrogram']['image_shape'] is None:
        # set where to determine the data path
        if dataset == 'dolphins':
            image_path = Path(os.path.join('outputs', 'spectrogram', 'val'))
        elif dataset == 'orcas':
            image_path = Path(os.path.join('outputs', cfg['paths']['pos_spec_path']))
        else:
            print(f"Processing this {dataset} dataset is not yet implemented")
        print(image_path)

        for root, _, files in os.walk(image_path):
            for file in files:
                if file.lower().endswith('jpg') or file.lower().endswith('png') or file.lower().endswith('jpeg'):
                    cfg['spectrogram']['image_shape'] = \
                        data_loader.DataLoader(cfg).get_image_shape(os.path.join(root, file))
                    print(f'Evaluating {os.path.join(root, file)} for image shape')
                    break
            # check that we have found a file for obtaining input shape, else continues
            if cfg['spectrogram']['image_shape'] is not None:
                break

    print(f"processing input shape of {cfg['spectrogram']['image_shape']}")

    # ----------------------------------------------------------------------------------------------------
    # Train model
    # ----------------------------------------------------------------------------------------------------
    if cfg['mode'] == 'train':
        train.train_model(cfg)

    # ----------------------------------------------------------------------------------------------------
    # Evaluate model
    # ----------------------------------------------------------------------------------------------------
    if cfg['mode'] == 'test':
        evaluate.predict_with_model(cfg)


if __name__ == "__main__":
    # Change these input parameters only if we are using different datasets than Dolphins
    main(download_from_s3=False,
         extract_file=False,  # if False, make sure metadata and wav files are in the right path under config
         dataset = 'dolphins')
