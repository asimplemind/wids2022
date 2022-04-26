"""test_aws_s3.py unittests for test_aws_s3.py
"""
import os
from pathlib import Path
import shutil
import unittest

from acoustic_ml import aws_s3

YOUR_BUCKET_NAME = ''
YOUR_FILE_URL = 'path/filename.tar.gz'
YOUR_FILE_NAME = 'filename.tar.gz'
YOUR_FOLDER = 'folder_name'


class TestAWSS3(unittest.TestCase):
    """AWS_S3 Unittest Class"""

    def test_download_s3_files(self):
        """Example for pulling data from aws"""
        if os.path.exists(os.path.join('tests', 'test_data`')):
            os.mkdir(os.path.join('tests', 'test_data'))

        # download successfully as of 15.03.2022
        s3_bucket_name = YOUR_BUCKET_NAME
        s3_url = YOUR_FILE_URL
        output_path = Path(os.path.join('tests', 'test_data'))
        s3 = aws_s3.AWSS3(s3_bucket_name, s3_url, output_path)
        s3.download_s3_files()
        self.assertIn(YOUR_FILE_NAME, os.listdir(output_path))

    def test_get_bucket_name(self):
        """Test getting bucket name"""
        if os.path.exists(os.path.join('tests', 'test_data`')):
            os.mkdir(os.path.join('tests', 'test_data'))

        s3_bucket_name = YOUR_BUCKET_NAME
        s3_url = YOUR_FILE_URL
        output_path = Path(os.path.join('tests', 'test_data'))
        s3 = aws_s3.AWSS3(s3_bucket_name, s3_url, output_path)
        bucket_name = s3.get_bucket_name()
        self.assertEqual(s3_bucket_name, bucket_name)

    def test_untar_file(self):
        """Test untaring file"""
        _ = aws_s3.untar_file(Path(os.path.join('tests', 'test_data', "untar_test.tar.gz")))
        self.assertIn("untar_test", os.listdir(os.path.join('tests', 'test_data', 'untar_data')))
        path_to_untared_folder = os.path.join('tests', 'test_data', 'untar_data', 'untar_test')
        shutil.rmtree(path_to_untared_folder)
