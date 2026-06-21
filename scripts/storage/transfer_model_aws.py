#!/usr/bin/env python3

"""Simple helper for reliably uploading very large files via multipart upload.

This script is designed to upload large files to S3 compatible APIs that are
backed by standard POSIX filesystems, which provide slower operations for
generating checksums and merging large files than standard Object Stores.
These slow operations can cause timeouts with typical clients and from proxies
operating between the client and the server. This script handles these timeouts
and retries accordingly.
Additionally, if a CompleteMultipartUpload request times out, the script will
wait and check to see if the upload did complete outside of the time out
threshold using HeadObject and comparing the size of the object to the local
file. This is particularly helpful when timeouts happen due to proxies or other
hops between the client and server terminating the connection, as increasing the
timeout threshold on client has no effect on such timeouts.

Environment variables can be used instead of command line flags for all
required credentials and most configuration values.

Based on https://github.com/runpod/runpod-s3-examples/blob/main/upload_large_file.py
"""

import argparse
import logging
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import boto3
from botocore.config import Config
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    ReadTimeoutError,
    ConnectTimeoutError,
)
from dotenv import load_dotenv

# Load .env from project root (three levels up from scripts/storage/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


# -----------------------------------------------------------------------------
# ─── SETUP ARGPARSE & ENV VARS ───────────────────────────────────────────────
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Multipart upload files to S3 with retries and logging. "
        "Credentials are loaded from .env automatically; CLI flags override them.",
    )
    parser.add_argument(
        "-b",
        "--bucket",
        default=os.environ.get("NETWORK_VOLUME_ID"),
        help="S3 bucket name (default: NETWORK_VOLUME_ID env var)",
    )
    parser.add_argument(
        "-c",
        "--chunk-size",
        type=int,
        default=100 * 1024 * 1024,
        help="Size of each chunk in bytes (default: 100 MB)",
    )
    # Single-file mode
    parser.add_argument(
        "-f",
        "--file",
        dest="file_path",
        help="Path of a single local file to upload (mutually exclusive with --source-dir)",
    )
    parser.add_argument(
        "-k",
        "--key",
        help="S3 object key for single-file mode (mutually exclusive with --source-dir)",
    )
    # Directory mode
    parser.add_argument(
        "-d",
        "--source-dir",
        default=str(_PROJECT_ROOT / "models" / "example_checkpoint"),
        help="Local directory to upload recursively (default: models/example_checkpoint). "
        "S3 keys are the file paths relative to the project root.",
    )
    parser.add_argument(
        "-a",
        "--access_key",
        default=os.environ.get("ACCESS_KEY") or os.environ.get("AWS_ACCESS_KEY_ID"),
        help="S3 access key (default: ACCESS_KEY or AWS_ACCESS_KEY_ID env var)",
    )
    parser.add_argument(
        "-s",
        "--secret_key",
        default=os.environ.get("SECRET_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY"),
        help="S3 secret key (default: SECRET_KEY or AWS_SECRET_ACCESS_KEY env var)",
    )
    parser.add_argument(
        "-e",
        "--endpoint",
        default=os.environ.get("ENDPOINT_URL") or os.environ.get("S3_ENDPOINT"),
        help="S3 endpoint URL (default: ENDPOINT_URL or S3_ENDPOINT env var)",
    )
    parser.add_argument(
        "-r",
        "--region",
        default=os.environ.get("DATACENTER") or os.environ.get("S3_REGION"),
        help="S3 region name (default: DATACENTER or S3_REGION env var)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only emit warnings and errors",
    )
    parser.add_argument(
        "-m",
        "--max-retries",
        type=int,
        default=int(os.environ.get("MAX_RETRIES", 5)),
        help="Maximum number of retries for each request (default: 5)",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# ─── SETUP LOGGING ───────────────────────────────────────────────────────────
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# ─── UPLOADER CLASS ─────────────────────────────────────────────────────────-
# -----------------------------------------------------------------------------
class LargeMultipartUploader:
    """Upload a large file using robust multipart uploads."""

    def __init__(
        self,
        *,
        file_path: str,
        bucket: str,
        key: str,
        region: str,
        access_key: str,
        secret_key: str,
        endpoint: str,
        part_size: int = 50 * 1024 * 1024,
        max_retries: int = 5,
    ) -> None:
        self.file_path = file_path
        self.bucket = bucket
        self.key = key
        self.region = region
        self.access_key = access_key
        self.secret_key = secret_key
        self.endpoint = endpoint
        self.part_size = part_size
        self.max_retries = max_retries

        self.progress_lock = Lock()
        self.parts_completed = 0

        self.session = boto3.session.Session(
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region,
        )
        self.botocore_cfg = Config(
            region_name=self.region,
            retries={"max_attempts": self.max_retries, "mode": "standard"},
        )
        self.s3 = self.session.client(
            "s3", config=self.botocore_cfg, endpoint_url=self.endpoint
        )
        self.upload_id: str | None = None

    @staticmethod
    def human_mb_per_s(num_bytes: int, seconds: float) -> float:
        """Return MB/s as float, avoiding divide-by-zero."""

        return (num_bytes / (1024 * 1024)) / seconds if seconds > 0 else float("inf")

    @staticmethod
    def is_insufficient_storage_error(exc: Exception) -> bool:
        """Return True if the exception wraps a 507 Insufficient Storage response."""

        if isinstance(exc, ClientError):
            meta = exc.response.get("ResponseMetadata", {})
            return meta.get("HTTPStatusCode") == 507
        return False

    @staticmethod
    def is_524_error(exc: Exception) -> bool:
        """Return True if the exception wraps a 524 timeout response."""

        if isinstance(exc, ClientError):
            meta = exc.response.get("ResponseMetadata", {})
            return meta.get("HTTPStatusCode") == 524
        return False

    @staticmethod
    def is_no_such_upload_error(exc: Exception) -> bool:
        """Return True if the exception reports a missing multipart upload."""

        if isinstance(exc, ClientError):
            err = exc.response.get("Error", {})
            return err.get("Code") == "NoSuchUpload"
        return False

    def call_with_524_retry(self, description: str, func):
        """Call ``func`` retrying on HTTP 524 or timeout errors."""

        for attempt in range(1, self.max_retries + 1):
            try:
                return func()
            except ClientError as exc:
                if self.is_524_error(exc):
                    logger.warning(
                        f"{description}: received 524 response (attempt {attempt})"
                    )
                    if attempt == self.max_retries:
                        logger.error(f"{description}: exceeded max_retries for 524")
                        raise
                    backoff = 2**attempt
                    logger.info(f"{description}: retrying in {backoff}s...")
                    time.sleep(backoff)
                    continue
                raise
            except (ReadTimeoutError, ConnectTimeoutError) as exc:
                logger.warning(
                    f"{description}: request timed out (attempt {attempt}): {exc}"
                )
                if attempt == self.max_retries:
                    logger.error(f"{description}: exceeded max_retries for timeout")
                    raise
                backoff = 2**attempt
                logger.info(f"{description}: retrying in {backoff}s...")
                time.sleep(backoff)

    def complete_with_timeout_retry(
        self,
        *,
        parts_sorted: list,
        initial_timeout: int,
        expected_size: int,
    ):
        """Complete the multipart upload, doubling timeout on client timeouts."""

        if self.upload_id is None:
            raise RuntimeError("upload_id not set")

        timeout = initial_timeout
        cfg = self.botocore_cfg
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            cfg = cfg.merge(Config(read_timeout=timeout, connect_timeout=timeout))
            client = self.session.client("s3", config=cfg, endpoint_url=self.endpoint)
            try:
                client.complete_multipart_upload(
                    Bucket=self.bucket,
                    Key=self.key,
                    UploadId=self.upload_id,
                    MultipartUpload={"Parts": parts_sorted},
                )
                self.s3 = client
                self.botocore_cfg = cfg
                return
            except (ReadTimeoutError, ConnectTimeoutError) as exc:
                last_exc = exc
                no_such_upload = False
                logger.warning(
                    f"complete_multipart_upload timed out after {timeout}s: {exc}"
                )
            except (ClientError, BotoCoreError) as exc:
                last_exc = exc
                no_such_upload = self.is_no_such_upload_error(exc)
                logger.warning(
                    f"complete_multipart_upload failed (attempt {attempt}): {exc}"
                )

            if no_such_upload:
                logger.info("Upload session missing; checking object state immediately")
            else:
                logger.info(
                    f"Waiting {timeout}s before checking object state to see if merge has completed"
                )
                time.sleep(timeout)

            try:
                head = self.call_with_524_retry(
                    "head_object",
                    lambda: client.head_object(Bucket=self.bucket, Key=self.key),
                )
                uploaded_size = head.get("ContentLength")
                if uploaded_size == expected_size:
                    logger.info(
                        "HeadObject confirms multipart upload merge has completed"
                    )
                    self.s3 = client
                    self.botocore_cfg = cfg
                    return
                logger.info(
                    "HeadObject size mismatch after timeout; will retry complete_multipart_upload"
                )
            except Exception as head_exc:
                logger.info(f"head_object failed after error: {head_exc}")

            if attempt == self.max_retries:
                raise (
                    last_exc
                    if last_exc
                    else RuntimeError(
                        "Exceeded max_retries without completing multipart upload"
                    )
                )

            timeout *= 2
            logger.info(f"Increasing timeout to {timeout}s and retrying")

    def upload_part(
        self,
        *,
        part_number: int,
        offset: int,
        bytes_to_read: int,
        total_parts: int,
        start_time: float,
    ) -> dict:
        """Upload a single part with exponential-backoff retries."""

        if self.upload_id is None:
            raise RuntimeError("upload_id not set")

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    f"Part {part_number}: reading {offset/1024**2:.1f}–{(offset+bytes_to_read)/1024**2:.1f} MB (attempt {attempt})"
                )
                with open(self.file_path, "rb") as f:
                    f.seek(offset)
                    data = f.read(bytes_to_read)
                resp = self.s3.upload_part(
                    Bucket=self.bucket,
                    Key=self.key,
                    PartNumber=part_number,
                    UploadId=self.upload_id,
                    Body=data,
                )
                etag = resp["ETag"]
                with self.progress_lock:
                    self.parts_completed += 1
                    progress = 100.0 * self.parts_completed / total_parts
                elapsed = time.time() - start_time
                progress_fraction = part_number / total_parts
                if progress_fraction > 0:
                    remaining = max(0, elapsed * (1 / progress_fraction - 1))
                    eta = time.strftime("%Hh %Mm %Ss", time.gmtime(remaining))
                else:
                    eta = "?"
                logger.info(
                    f"Part {part_number}: uploaded, progress: {progress:.1f}%, est time remaining: {eta}"
                )
                return {"PartNumber": part_number, "ETag": etag}
            except (BotoCoreError, ClientError) as exc:
                if self.is_insufficient_storage_error(exc):
                    logger.error(
                        f"Part {part_number}: received 507 Insufficient Storage; aborting"
                    )
                    raise RuntimeError("Server reported insufficient storage") from exc
                if self.is_524_error(exc):
                    logger.warning(
                        f"Part {part_number}: received 524 response (attempt {attempt})"
                    )
                else:
                    logger.warning(
                        f"Part {part_number}: attempt {attempt} failed: {exc}"
                    )
                if attempt == self.max_retries:
                    logger.error(
                        f"Part {part_number}: exceeded max_retries ({self.max_retries})"
                    )
                    raise
                backoff = 2**attempt
                logger.info(f"Part {part_number}: retrying in {backoff}s...")
                time.sleep(backoff)

    # ------------------------------------------------------------------
    # Main upload driver
    # ------------------------------------------------------------------
    def upload(self) -> None:
        """Execute the multipart upload."""

        logger.info(
            f"Uploading to region: {self.region}; bucket: {self.bucket}; key: {self.key}"
        )

        file_size = os.path.getsize(self.file_path)
        total_parts = math.ceil(file_size / self.part_size)
        logger.info(
            f"File size: {file_size/1024**2:.1f} MB; will upload in {total_parts} parts of up to {self.part_size/1024**2:.0f} MB each"
        )

        start_time = time.time()

        file_gb = file_size / float(1024**3)
        completion_timeout = max(60, int(math.ceil(file_gb) * 5))

        resp = self.call_with_524_retry(
            "create_multipart_upload",
            lambda: self.s3.create_multipart_upload(Bucket=self.bucket, Key=self.key),
        )
        self.upload_id = resp["UploadId"]
        logger.info(f"Initiated multipart upload: UploadId={self.upload_id}")

        parts: list[dict] = []
        try:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {}
                for part_num in range(1, total_parts + 1):
                    offset = (part_num - 1) * self.part_size
                    chunk_size = min(self.part_size, file_size - offset)
                    futures[
                        executor.submit(
                            self.upload_part,
                            part_number=part_num,
                            offset=offset,
                            bytes_to_read=chunk_size,
                            total_parts=total_parts,
                            start_time=start_time,
                        )
                    ] = part_num

                for fut in as_completed(futures):
                    part = fut.result()
                    parts.append(part)

            def fetch_parts():
                paginator = self.s3.get_paginator("list_parts")
                found = []
                for page in paginator.paginate(
                    Bucket=self.bucket, Key=self.key, UploadId=self.upload_id
                ):
                    found.extend(page.get("Parts", []))
                return found

            seen = self.call_with_524_retry("list_parts", fetch_parts)
            logger.info(f"Verified {len(seen)} of {total_parts} parts uploaded")

            if len(seen) != total_parts:
                raise RuntimeError(f"Expected {total_parts} parts but saw {len(seen)}")

            parts_sorted = sorted(parts, key=lambda x: x["PartNumber"])
            logger.info("Sending complete_multipart_upload request")
            self.complete_with_timeout_retry(
                parts_sorted=parts_sorted,
                initial_timeout=completion_timeout,
                expected_size=file_size,
            )

            head = self.call_with_524_retry(
                "head_object",
                lambda: self.s3.head_object(Bucket=self.bucket, Key=self.key),
            )
            uploaded_size = head.get("ContentLength")
            if uploaded_size != file_size:
                logger.error(
                    f"Size mismatch: remote object is {uploaded_size/1024**2:.1f} MB, "
                    f"but local file is {file_size/1024**2:.1f} MB"
                )
                raise RuntimeError(
                    "Multipart upload verification failed: size mismatch"
                )
            logger.info(
                f"Verified upload: remote object size {uploaded_size/1024**2:.1f} MB matches local file size"
            )
        except Exception as exc:
            logger.error(f"Upload interrupted: {exc}")
            if self.upload_id:
                logger.info(f"UploadId {self.upload_id} left open for resumption")
            raise

        elapsed = time.time() - start_time
        speed = self.human_mb_per_s(file_size, elapsed)
        duration = time.strftime("%Hh %Mm %Ss", time.gmtime(elapsed))
        logger.info(f"Upload Speed {speed:.2f} MB/s, Duration {duration}")


def upload_directory(
    source_dir: str,
    *,
    bucket: str,
    access_key: str,
    secret_key: str,
    endpoint: str,
    region: str,
    part_size: int,
    max_retries: int,
) -> None:
    """Upload every file under source_dir to S3.

    S3 keys are the file paths relative to the project root, so
    models/example_checkpoint/global_step75_hf/config.json in the repo
    becomes that same path as the S3 object key.
    """
    source_path = Path(source_dir).resolve()
    if not source_path.is_dir():
        raise ValueError(f"source-dir does not exist or is not a directory: {source_path}")

    files = sorted(f for f in source_path.rglob("*") if f.is_file())
    if not files:
        logger.warning(f"No files found under {source_path}")
        return

    logger.info(f"Uploading {len(files)} file(s) from {source_path} → s3://{bucket}/")

    for i, file_path in enumerate(files, 1):
        s3_key = str(file_path.relative_to(_PROJECT_ROOT))
        logger.info(f"[{i}/{len(files)}] {s3_key}")
        LargeMultipartUploader(
            file_path=str(file_path),
            bucket=bucket,
            key=s3_key,
            region=region,
            access_key=access_key,
            secret_key=secret_key,
            endpoint=endpoint,
            part_size=part_size,
            max_retries=max_retries,
        ).upload()

    logger.info(f"All {len(files)} file(s) uploaded successfully.")


if __name__ == "__main__":
    args = parse_args()
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    missing = [n for n, v in [
        ("bucket (NETWORK_VOLUME_ID)", args.bucket),
        ("access_key (ACCESS_KEY)", args.access_key),
        ("secret_key (SECRET_KEY)", args.secret_key),
        ("endpoint (ENDPOINT_URL)", args.endpoint),
        ("region (DATACENTER)", args.region),
    ] if not v]
    if missing:
        logger.error("Missing required credentials/config: " + ", ".join(missing))
        sys.exit(1)

    if args.file_path:
        # Single-file mode: --file and --key are both required
        if not args.key:
            logger.error("--key is required when using --file")
            sys.exit(1)
        LargeMultipartUploader(
            file_path=args.file_path,
            bucket=args.bucket,
            key=args.key,
            region=args.region,
            access_key=args.access_key,
            secret_key=args.secret_key,
            endpoint=args.endpoint,
            part_size=args.chunk_size,
            max_retries=args.max_retries,
        ).upload()
    else:
        # Directory mode (default)
        upload_directory(
            args.source_dir,
            bucket=args.bucket,
            access_key=args.access_key,
            secret_key=args.secret_key,
            endpoint=args.endpoint,
            region=args.region,
            part_size=args.chunk_size,
            max_retries=args.max_retries,
        )