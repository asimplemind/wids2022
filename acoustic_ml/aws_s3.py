"""This class handles APIs with S3"""
import os
import sys
from pathlib import Path
import tarfile

import boto3
import botocore
from botocore.config import Config


class AWSS3:
    """Collection of functions to access data from Amazon Simple Storage Service (S3)"""

    def __init__(self, bucket_name: str, url: str, out_path: Path):
        """constructor"""
        self.bucket_name = bucket_name
        self.url = url
        self.output_path = out_path
        self.s3_resource = boto3.resource('s3', config=Config(signature_version=botocore.UNSIGNED))

    def download_s3_files(self):
        """Get list of files from s3 bucket and download data/meta files"""

        s3_bucket = self.s3_resource.Bucket(self.bucket_name)
        s3_files = s3_bucket.objects.filter(Prefix=self.url).all()
        for file in s3_files:
            if file.key.lower().endswith(".gz") or file.key.lower().endswith(".tsv"):
                filepath, filename = os.path.split(file.key)

                if not os.path.exists(os.path.join(self.output_path, filepath)):
                    Path(self.output_path).mkdir(parents=True, exist_ok=True)

                print("Downloading file -- {}".format(file.key))
                try:
                    s3_bucket.download_file(file.key, os.path.join(self.output_path, filename))
                except IOError as e:
                    print(f"Error downloading from s3 bucket {e}")

    def get_bucket_name(self):
        """Return S3's bucket name"""
        return self.bucket_name


def untar_file(tar_file: Path) -> Path:
    """Extract tar.gz file

    Arguments:
        tar_file (Path): full path and filename ending with .tar or .tar.gz
    Returns:
        (Path): location where files were untar to
    """
    try:
        with tarfile.open(tar_file, 'r') as t:
            print(f"Saving extract tar file to {os.path.join(os.path.dirname(tar_file), 'untar_data')}")
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(t, os.path.join(os.path.dirname(tar_file),"untar_data"))
            print(os.listdir(os.path.join(os.path.dirname(tar_file), 'untar_data')))
    except tarfile.ReadError as e:
        print(f"Error extracting tar file {e}")
        sys.exit()

    print(f"File {os.path.basename(tar_file)} extracted successfully")

    return Path(os.path.join(os.path.dirname(tar_file), 'untar_data',
                             os.listdir(os.path.join(os.path.dirname(tar_file), 'untar_data'))[0]))
