"""
upload_to_runpod.py
-------------------
Uploads extracted_clips.zip to the RunPod Network Volume via S3-compatible API.

Usage:
    python student_training/scripts/upload_to_runpod.py --api_key YOUR_RUNPOD_API_KEY

Get your API key: RunPod → Settings → API Keys
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ZIP_FILE     = PROJECT_ROOT / "outputs" / "extracted_clips.zip"

BUCKET       = "0hnvco2s4j"
ENDPOINT_URL = "https://s3api-eu-ro-1.runpod.io"
REGION       = "eu-ro-1"
OBJECT_KEY   = "extracted_clips.zip"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", required=True, help="S3 Access Key ID")
    parser.add_argument("--secret_key", required=True, help="S3 Secret Access Key")
    parser.add_argument("--zip_file", default=str(ZIP_FILE), help="Path to zip file")
    args = parser.parse_args()

    zip_path = Path(args.zip_file)
    if not zip_path.exists():
        print(f"[ERROR] Zip file not found: {zip_path}")
        sys.exit(1)

    file_size_mb = zip_path.stat().st_size / 1024 / 1024
    print(f"File     : {zip_path}")
    print(f"Size     : {file_size_mb:.1f} MB")
    print(f"Bucket   : {BUCKET}")
    print(f"Endpoint : {ENDPOINT_URL}")
    print()

    try:
        import boto3
        from boto3.s3.transfer import TransferConfig
    except ImportError:
        print("[ERROR] boto3 not installed. Run:")
        print("  pip install boto3")
        sys.exit(1)

    s3 = boto3.client(
        "s3",
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=args.api_key,
        aws_secret_access_key=args.secret_key,
        region_name=REGION,
    )

    # Multipart upload with progress callback
    uploaded_bytes = [0]
    total_bytes = zip_path.stat().st_size

    def progress(chunk):
        uploaded_bytes[0] += chunk
        pct = uploaded_bytes[0] / total_bytes * 100
        mb_done = uploaded_bytes[0] / 1024 / 1024
        print(f"\r  Uploading... {mb_done:.1f}/{file_size_mb:.1f} MB ({pct:.1f}%)", end="", flush=True)

    config = TransferConfig(multipart_threshold=10 * 1024 * 1024,  # 10MB chunks
                            max_concurrency=4)

    print(f"Uploading to s3://{BUCKET}/{OBJECT_KEY} ...")
    s3.upload_file(
        str(zip_path),
        BUCKET,
        OBJECT_KEY,
        Callback=progress,
        Config=config,
    )
    print(f"\n\nUpload complete!")
    print(f"Verify with:")
    print(f"  aws s3 ls --region {REGION} --endpoint-url {ENDPOINT_URL} s3://{BUCKET}/")


if __name__ == "__main__":
    main()
