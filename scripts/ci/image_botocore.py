"""PRD-262 US-021: botocore in the API image carries only the services we call, and they work.

orchestrator/Dockerfile's pybuild stage deletes every service model under
``botocore/data`` except KEPT_SERVICES; the top-level files stay. CI runs this inside the
built image, with the image's own Python and as the image's user:

    docker run --rm -i --entrypoint python IMAGE - < scripts/ci/image_botocore.py

It checks that:

- ``botocore/data`` holds exactly the KEPT_SERVICES directories, and TOP_LEVEL_FILES;
- boto3 builds a client for each kept service, which loads the service's model and its
  endpoint rules. Throwaway keys and an explicit region mean no credential lookup and
  no network call. The s3 client is configured for SigV4, as core/storage/s3.py
  configures the app's, and presigns a GET, which is signing done offline;
- a service outside the list (PRUNED_PROBE) raises UnknownServiceError, the error any
  other client would now raise.

orchestrator/tests/test_prd262_botocore_services.py holds KEPT_SERVICES to the
Dockerfile's list and fails when code in orchestrator/ names a service outside it.

Exit 0 when every check passes, 1 otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Sequence

KEPT_SERVICES = ("s3", "s3vectors", "bedrock-runtime", "sts", "sso", "sso-oidc")
TOP_LEVEL_FILES = ("_retry.json", "endpoints.json", "partitions.json", "sdk-default-configuration.json")
PRUNED_PROBE = "dynamodb"  # a service no code calls: its model is gone
PROBE_REGION = "us-east-1"
PROBE_KEY = "probe"  # throwaway, so boto3 never walks the credential chain
PRESIGN_SECONDS = 60
APP_SIGNATURE_VERSION = "v4"  # core/storage/s3.py's SIGNATURE_VERSION, which every S3 client uses
SIGNATURE_PARAM = "X-Amz-Signature="  # a SigV4 presign; SigV2's carries Signature= alone
ERROR_TITLE = "botocore (PRD-262 US-021)"


def _described(error: BaseException) -> str:
    """``Type: message`` on one line, so it fits a workflow annotation. Pure."""
    return f"{type(error).__name__}: {' '.join(str(error).split())}"


def data_problems(data: Path) -> List[str]:
    """A service directory the prune left, or a kept directory or top-level file it lost."""
    if not data.is_dir():
        return [f"{data} is not a directory"]
    services = sorted(path.name for path in data.iterdir() if path.is_dir())
    problems = [f"botocore/data still has {name}/" for name in services if name not in KEPT_SERVICES]
    problems += [f"botocore/data lost {name}/, a service we call" for name in KEPT_SERVICES if name not in services]
    problems += [f"botocore/data lost {name}" for name in TOP_LEVEL_FILES if not (data / name).is_file()]
    return problems


def _session(boto3_module):
    return boto3_module.session.Session(
        aws_access_key_id=PROBE_KEY, aws_secret_access_key=PROBE_KEY, region_name=PROBE_REGION)


def _presign_problems(client) -> List[str]:
    """An s3 client signs a GET offline: the model, the endpoint rules and SigV4 all work."""
    url = client.generate_presigned_url(
        "get_object", Params={"Bucket": PROBE_KEY, "Key": PROBE_KEY}, ExpiresIn=PRESIGN_SECONDS)
    return [] if SIGNATURE_PARAM in url else [f"the s3 client presigned {url} with no signature"]


def client_problems(session, services: Sequence[str], s3_config) -> List[str]:
    """Build a client for each of ``services``; the s3 one, with ``s3_config``, presigns a GET too."""
    problems = []
    for name in services:
        try:
            client = session.client(name, config=s3_config if name == "s3" else None)
            print(f"{name}: model {client.meta.service_model.api_version}, endpoint {client.meta.endpoint_url}")
            problems += _presign_problems(client) if name == "s3" else []
        except Exception as error:  # noqa: BLE001 — any failure is this service's, reported by name
            problems.append(f"boto3 can't build a {name} client: {_described(error)}")
    return problems


def pruned_problems(session, unknown_error: type) -> List[str]:
    """PRUNED_PROBE must be unknown to boto3: the prune removed its model."""
    try:
        session.client(PRUNED_PROBE)
    except unknown_error:
        return []
    except Exception as error:  # noqa: BLE001 — anything but UnknownServiceError is a wrong answer
        return [f"a {PRUNED_PROBE} client failed with {_described(error)}, not UnknownServiceError"]
    return [f"boto3 still builds a {PRUNED_PROBE} client: the prune left its model"]


def main() -> int:
    sys.path = [entry for entry in sys.path if entry]  # stdin's '' is the working directory, /app
    import boto3
    import botocore
    from botocore.config import Config
    from botocore.exceptions import UnknownServiceError

    data = Path(botocore.__file__).parent / "data"
    print(f"botocore {botocore.__version__}, boto3 {boto3.__version__}, data at {data}")
    session = _session(boto3)
    problems = data_problems(data)
    problems += client_problems(session, KEPT_SERVICES, Config(signature_version=APP_SIGNATURE_VERSION))
    problems += pruned_problems(session, UnknownServiceError)
    for problem in problems:
        print(f"::error title={ERROR_TITLE}::{problem}")
    if not problems:
        print(f"botocore/data holds {', '.join(KEPT_SERVICES)} and its top-level files, boto3 builds each "
              f"client, and a {PRUNED_PROBE} client is unknown")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
