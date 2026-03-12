"""Deploy (or tear down) a SageMaker real-time endpoint for MedGemma."""

from __future__ import annotations

import argparse
import os
import re
import sys

import boto3  # type: ignore[import-untyped]
from botocore.client import BaseClient  # type: ignore[import-untyped]
from botocore.exceptions import ClientError, NoCredentialsError  # type: ignore[import-untyped]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Deploy or tear down a SageMaker real-time endpoint for MedGemma."
    )
    parser.add_argument(
        "--model-data",
        required=True,
        help="S3 URI to model.tar.gz (e.g. s3://bucket/medgemma/model.tar.gz)",
    )
    parser.add_argument(
        "--role-arn",
        default=None,
        help="SageMaker execution role ARN (default: $SAGEMAKER_ROLE_ARN)",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="AWS region (default: boto3 default)",
    )
    parser.add_argument(
        "--instance-type",
        default="ml.g5.xlarge",
        help="Instance type (default: ml.g5.xlarge, 24 GB VRAM)",
    )
    parser.add_argument(
        "--endpoint-name",
        default="medgemma-endpoint",
        help="Endpoint name (default: medgemma-endpoint)",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token for gated tokenizer refs (default: $HF_TOKEN or $MED_GEM_TOKEN)",
    )
    parser.add_argument(
        "--image-uri",
        default=None,
        help="Override DLC image URI (bypasses SDK version resolution)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved config and exit without executing",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete the endpoint instead of creating it",
    )
    return parser.parse_args(argv)


def get_role_arn(cli_role: str | None) -> str:
    """Resolve the SageMaker execution role from CLI arg or environment variable."""
    role = cli_role or os.environ.get("SAGEMAKER_ROLE_ARN")
    if not role:
        print(
            "Error: No SageMaker execution role ARN provided.\n"
            "Set SAGEMAKER_ROLE_ARN environment variable or pass --role-arn.",
            file=sys.stderr,
        )
        sys.exit(1)
    return role


def get_hf_token(cli_token: str | None) -> str | None:
    """Resolve the HuggingFace token from CLI arg or environment variables."""
    return cli_token or os.environ.get("HF_TOKEN") or os.environ.get("MED_GEM_TOKEN")


def validate_s3_object(s3_uri: str, session: boto3.Session) -> None:
    """Verify that the S3 object at the given URI exists."""
    match = re.match(r"^s3://([^/]+)/(.+)$", s3_uri)
    if not match:
        print(f"Error: Invalid S3 URI: {s3_uri}", file=sys.stderr)
        sys.exit(1)

    bucket, key = match.group(1), match.group(2)
    s3_client = session.client("s3")
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code in ("404", "NoSuchKey"):
            print(
                f"Error: S3 object not found: {s3_uri}\n"
                f"Bucket: {bucket}, Key: {key}",
                file=sys.stderr,
            )
        else:
            print(f"Error checking S3 object: {exc}", file=sys.stderr)
        sys.exit(1)


def delete_endpoint(endpoint_name: str, sm_client: BaseClient) -> None:
    """Delete a SageMaker endpoint, its config, and its model."""
    # Delete endpoint
    try:
        sm_client.delete_endpoint(EndpointName=endpoint_name)
        print(f"Deleted endpoint: {endpoint_name}")
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "ValidationException":
            print(f"Endpoint not found: {endpoint_name}")
        else:
            raise

    # Delete endpoint config
    try:
        sm_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
        print(f"Deleted endpoint config: {endpoint_name}")
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "ValidationException":
            print(f"Endpoint config not found: {endpoint_name}")
        else:
            raise

    # Delete model
    try:
        sm_client.delete_model(ModelName=endpoint_name)
        print(f"Deleted model: {endpoint_name}")
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "ValidationException":
            print(f"Model not found: {endpoint_name}")
        else:
            raise


DLC_TRANSFORMERS_VERSION = "4.51.3"
DLC_PYTORCH_VERSION = "2.6.0"
DLC_PY_VERSION = "py312"


def resolve_image_uri(region: str, instance_type: str) -> str:
    """Resolve the HuggingFace DLC image URI via the SageMaker SDK."""
    from sagemaker.core.image_uris import retrieve  # type: ignore[import-untyped]

    try:
        return retrieve(  # type: ignore[no-any-return]
            framework="huggingface",
            region=region,
            version=DLC_TRANSFORMERS_VERSION,
            base_framework_version=f"pytorch{DLC_PYTORCH_VERSION}",
            py_version=DLC_PY_VERSION,
            instance_type=instance_type,
            image_scope="inference",
        )
    except ValueError as exc:
        print(
            f"Error: DLC image resolution failed.\n{exc}\n"
            "Try passing --image-uri with a specific container image URI.",
            file=sys.stderr,
        )
        sys.exit(1)


def deploy_model(
    model_data: str,
    role_arn: str,
    instance_type: str,
    endpoint_name: str,
    region: str,
    hf_token: str | None,
    image_uri: str | None,
    sm_client: BaseClient | None = None,
) -> str:
    """Create a SageMaker model, endpoint config, and endpoint using the boto3 API."""
    if sm_client is None:
        sm_client = boto3.client("sagemaker", region_name=region)

    if not image_uri:
        image_uri = resolve_image_uri(region, instance_type)
        print(f"Resolved DLC image: {image_uri}")

    env: dict[str, str] = {}
    if hf_token:
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token

    container = {
        "Image": image_uri,
        "ModelDataUrl": model_data,
        "Environment": env,
    }

    print(f"Deploying to endpoint '{endpoint_name}' on {instance_type} ...")
    print("This typically takes 5-15 minutes.")

    # Create model
    sm_client.create_model(
        ModelName=endpoint_name,
        PrimaryContainer=container,
        ExecutionRoleArn=role_arn,
    )
    print(f"Created model: {endpoint_name}")

    # Create endpoint config
    sm_client.create_endpoint_config(
        EndpointConfigName=endpoint_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": endpoint_name,
                "InstanceType": instance_type,
                "InitialInstanceCount": 1,
            }
        ],
    )
    print(f"Created endpoint config: {endpoint_name}")

    # Create endpoint and wait for it to be in service
    sm_client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_name,
    )
    print(f"Created endpoint: {endpoint_name} (waiting for InService ...)")

    waiter = sm_client.get_waiter("endpoint_in_service")
    try:
        waiter.wait(
            EndpointName=endpoint_name,
            WaiterConfig={"Delay": 30, "MaxAttempts": 60},
        )
    except Exception as exc:
        print(
            f"Error: Endpoint did not reach InService status.\n{exc}\n"
            "Check CloudWatch logs for container startup errors:\n"
            f"  /aws/sagemaker/Endpoints/{endpoint_name}",
            file=sys.stderr,
        )
        sys.exit(1)

    return endpoint_name


def main(argv: list[str] | None = None) -> None:
    """Deploy or tear down a SageMaker endpoint for MedGemma."""
    args = parse_args(argv)

    # Resolve region and create session
    session_kwargs: dict[str, str] = {}
    if args.region:
        session_kwargs["region_name"] = args.region

    try:
        session = boto3.Session(**session_kwargs)
    except NoCredentialsError:
        print(
            "Error: No AWS credentials found.\n"
            "Configure via AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, "
            "~/.aws/credentials, or an IAM role.",
            file=sys.stderr,
        )
        sys.exit(1)

    region = session.region_name
    if not region:
        print(
            "Error: No AWS region configured.\n"
            "Set AWS_DEFAULT_REGION environment variable or pass --region.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate credentials
    try:
        sts = session.client("sts")
        identity = sts.get_caller_identity()
        print(f"AWS account: {identity['Account']}, region: {region}")
    except NoCredentialsError:
        print(
            "Error: No AWS credentials found.\n"
            "Configure via AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, "
            "~/.aws/credentials, or an IAM role.",
            file=sys.stderr,
        )
        sys.exit(1)
    except ClientError as exc:
        print(f"Error: AWS credential check failed: {exc}", file=sys.stderr)
        sys.exit(1)

    # Handle --delete
    if args.delete:
        sm_client = session.client("sagemaker")
        delete_endpoint(args.endpoint_name, sm_client)
        print("\nTeardown complete.")
        return

    # Resolve required config for deploy
    role_arn = get_role_arn(args.role_arn)

    # Handle --dry-run
    if args.dry_run:
        hf_token = get_hf_token(args.hf_token)
        print("\n--- Dry-run config ---")
        print(f"  Model data:     {args.model_data}")
        print(f"  Role ARN:       {role_arn}")
        print(f"  Region:         {region}")
        print(f"  Instance type:  {args.instance_type}")
        print(f"  Endpoint name:  {args.endpoint_name}")
        print(f"  Image URI:      {args.image_uri or '(SDK-resolved DLC)'}")
        print(f"  HF token:       {'***' if hf_token else '(not set)'}")
        return

    # Validate S3 object exists
    validate_s3_object(args.model_data, session)

    # Check endpoint doesn't already exist
    sm_client = session.client("sagemaker")
    try:
        sm_client.describe_endpoint(EndpointName=args.endpoint_name)
        print(
            f"Error: Endpoint '{args.endpoint_name}' already exists.\n"
            f"To tear it down first, run:\n"
            f"  python src/deploy.py --delete --endpoint-name {args.endpoint_name}",
            file=sys.stderr,
        )
        sys.exit(1)
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ValidationException":
            raise
        # ValidationException means endpoint doesn't exist — expected

    # Deploy
    hf_token = get_hf_token(args.hf_token)
    endpoint = deploy_model(
        model_data=args.model_data,
        role_arn=role_arn,
        instance_type=args.instance_type,
        endpoint_name=args.endpoint_name,
        region=region,
        hf_token=hf_token,
        image_uri=args.image_uri,
        sm_client=sm_client,
    )

    print(f"\nEndpoint '{endpoint}' is InService.")
    print("\nTest with:")
    print(
        f"  aws sagemaker-runtime invoke-endpoint \\\n"
        f"    --endpoint-name {endpoint} \\\n"
        f"    --content-type application/json \\\n"
        f"    --cli-binary-format raw-in-base64-out \\\n"
        f"    --body '{{\"text\": \"What are symptoms of pneumonia?\"}}' \\\n"
        f"    /dev/stdout"
    )


if __name__ == "__main__":
    main()
