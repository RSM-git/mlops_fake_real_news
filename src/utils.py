import yaml
from google.cloud import storage
from google.oauth2 import service_account
from yaml.loader import SafeLoader


def load_yaml(file_path: str) -> dict:
    """safely loads a yaml file and returns
    it as a dict

    Args:
        file_path (str): path to the yaml file with extension

    Returns:
        dict: key-value pairs of the yaml
    """
    with open(file_path, "r") as f:
        data = yaml.load(f, Loader=SafeLoader)
    return data


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # code taken from gcloud official documentation
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # The ID of your GCS object
    # source_blob_name = "storage-object-name"

    # The path to which the file should be downloaded
    # destination_file_name = "local/path/to/file"

    credentials = service_account.Credentials.from_service_account_file(
        "corded-pivot-374409-d64b3d69468f.json"
    )

    storage_client = storage.Client(credentials=credentials)

    bucket = storage_client.bucket(bucket_name)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Downloaded storage object {} from bucket {} to local file {}.".format(
            source_blob_name, bucket_name, destination_file_name
        )
    )


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # code taken from gcloud official documentation
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    credentials = service_account.Credentials.from_service_account_file(
        "corded-pivot-374409-dbccae470422.json"
    )

    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(f"File {source_file_name} uploaded to {destination_blob_name}.")
