import os
import boto3
import pandas as pd


class S3Wrap(object):
    def __init__(self, s3_resource):
        self.s3_resource = s3_resource

    def load_dataframe(self, bucket, file_path_s3):
        file = self.s3_resource.Object(bucket, file_path_s3)
        file = file.get()
        return pd.read_csv(file['Body'])

    def copy_between_buckets(self, from_bucket, from_path, target_bucket, target_path):
        self.s3_resource.Object(target_bucket, target_path).copy_from(
            CopySource=os.path.join(from_bucket, from_path)
        )

    def save_dataframe(self, dataframe, bucket, file_name_s3):
        dfstring = dataframe.to_csv(None)
        self.s3_resource.Object(bucket, file_name_s3).put(Body=dfstring)

    def download(self, bucket, path_s3, path_local):
        bucket = self.s3_resource.Bucket(bucket)
        bucket.download_file(path_s3, path_local)

    def upload(self, bucket, path_local, path_s3):
        bucket = self.s3_resource.Bucket(bucket)
        bucket.upload_file(path_local, path_s3)

    def list_all_files(self, bucket):
        bucket = self.s3_resource.Bucket(bucket)
        return bucket.objects.all()

    def check_if_exists(self, bucket, key):
        bucket = self.s3_resource.Bucket(bucket)
        return bool(list(bucket.objects.filter(Prefix=key)))

    def wait_for_file(self, bucket, key):
        bucket = self.s3_resource.Bucket(bucket)
        obj = bucket.Object(key)
        obj.wait_until_exists()

    @staticmethod
    def download_anon(bucket, path_s3, path_local):
        s3_resource = boto3.resource('s3')
        s3_resource.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)
        bucket = s3_resource.Bucket(bucket)
        bucket.download_file(path_s3, path_local)


# class S3Checkpoint(keras.callbacks.Callback, S3Wrap):
#     def __init__(self, s3_resource, bucket, path_local, path_s3):
#         self.bucket = bucket
#         self.path_local = path_local
#         self.path_s3 = path_s3
#         self.last_change = None
#
#         S3Wrap.__init__(self, s3_resource)
#
#     def on_epoch_end(self, *args):
#         epoch_nr, logs = args
#
#         if os.path.getmtime(self.path_local) != self.last_change:
#             self.upload(self.bucket, self.path_local, self.path_s3 + str(os.path.getmtime(self.path_local)) + ".h5")
#             print('uploading checkpoint')
#             self.last_change = os.path.getmtime(self.path_local)
#         else:
#             print("model didn't improve - no upload")
