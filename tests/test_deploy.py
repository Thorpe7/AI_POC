"""Unit tests for src/deploy.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError, NoCredentialsError

from deploy import (
    delete_endpoint,
    deploy_model,
    get_hf_token,
    get_role_arn,
    main,
    parse_args,
    resolve_image_uri,
    validate_s3_object,
)

# --- parse_args ---


class TestParseArgs:
    def test_required_model_data(self) -> None:
        with pytest.raises(SystemExit):
            parse_args([])

    def test_required_endpoint_name(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["--model-data", "s3://bucket/model.tar.gz"])

    def test_defaults(self) -> None:
        args = parse_args([
            "--model-data", "s3://bucket/model.tar.gz",
            "--endpoint-name", "my-endpoint",
        ])
        assert args.model_data == "s3://bucket/model.tar.gz"
        assert args.role_arn is None
        assert args.region is None
        assert args.instance_type == "ml.g5.xlarge"
        assert args.endpoint_name == "my-endpoint"
        assert args.hf_token is None
        assert args.image_uri is None
        assert args.dry_run is False
        assert args.delete is False
        assert args.async_inference is False
        assert args.async_s3_output is None

    def test_all_flags(self) -> None:
        args = parse_args([
            "--model-data", "s3://b/m.tar.gz",
            "--role-arn", "arn:aws:iam::123:role/R",
            "--region", "us-west-2",
            "--instance-type", "ml.g5.2xlarge",
            "--endpoint-name", "my-ep",
            "--hf-token", "hf_abc",
            "--image-uri", "123.dkr.ecr.us-east-1.amazonaws.com/img:tag",
            "--dry-run",
            "--delete",
        ])
        assert args.model_data == "s3://b/m.tar.gz"
        assert args.role_arn == "arn:aws:iam::123:role/R"
        assert args.region == "us-west-2"
        assert args.instance_type == "ml.g5.2xlarge"
        assert args.endpoint_name == "my-ep"
        assert args.hf_token == "hf_abc"
        assert args.image_uri == "123.dkr.ecr.us-east-1.amazonaws.com/img:tag"
        assert args.dry_run is True
        assert args.delete is True


# --- get_role_arn ---


class TestGetRoleArn:
    def test_from_cli(self) -> None:
        assert get_role_arn("arn:aws:iam::123:role/R") == "arn:aws:iam::123:role/R"

    def test_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SAGEMAKER_ROLE_ARN", "arn:aws:iam::456:role/S")
        assert get_role_arn(None) == "arn:aws:iam::456:role/S"

    def test_cli_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SAGEMAKER_ROLE_ARN", "arn:env")
        assert get_role_arn("arn:cli") == "arn:cli"

    def test_missing_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SAGEMAKER_ROLE_ARN", raising=False)
        with pytest.raises(SystemExit):
            get_role_arn(None)


# --- get_hf_token ---


class TestGetHfToken:
    def test_from_cli(self) -> None:
        assert get_hf_token("hf_abc") == "hf_abc"

    def test_from_hf_token_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HF_TOKEN", "hf_env")
        assert get_hf_token(None) == "hf_env"

    def test_from_med_gem_token_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setenv("MED_GEM_TOKEN", "mg_env")
        assert get_hf_token(None) == "mg_env"

    def test_hf_token_takes_precedence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HF_TOKEN", "hf_first")
        monkeypatch.setenv("MED_GEM_TOKEN", "mg_second")
        assert get_hf_token(None) == "hf_first"

    def test_returns_none_when_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("MED_GEM_TOKEN", raising=False)
        assert get_hf_token(None) is None


# --- validate_s3_object ---


class TestValidateS3Object:
    def test_valid_uri_and_object_exists(self) -> None:
        session = MagicMock()
        s3 = MagicMock()
        session.client.return_value = s3
        s3.head_object.return_value = {}

        validate_s3_object("s3://my-bucket/path/model.tar.gz", session)
        s3.head_object.assert_called_once_with(Bucket="my-bucket", Key="path/model.tar.gz")

    def test_invalid_uri_exits(self) -> None:
        session = MagicMock()
        with pytest.raises(SystemExit):
            validate_s3_object("not-an-s3-uri", session)

    def test_object_not_found_exits(self) -> None:
        session = MagicMock()
        s3 = MagicMock()
        session.client.return_value = s3
        s3.head_object.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
        )
        with pytest.raises(SystemExit):
            validate_s3_object("s3://bucket/missing.tar.gz", session)

    def test_other_client_error_exits(self) -> None:
        session = MagicMock()
        s3 = MagicMock()
        session.client.return_value = s3
        s3.head_object.side_effect = ClientError(
            {"Error": {"Code": "403", "Message": "Forbidden"}}, "HeadObject"
        )
        with pytest.raises(SystemExit):
            validate_s3_object("s3://bucket/model.tar.gz", session)


# --- delete_endpoint ---


class TestDeleteEndpoint:
    def test_deletes_all_three_resources(self) -> None:
        sm = MagicMock()
        delete_endpoint("my-ep", sm)
        sm.delete_endpoint.assert_called_once_with(EndpointName="my-ep")
        sm.delete_endpoint_config.assert_called_once_with(EndpointConfigName="my-ep")
        sm.delete_model.assert_called_once_with(ModelName="my-ep")

    def test_handles_missing_resources(self) -> None:
        sm = MagicMock()
        not_found = ClientError(
            {"Error": {"Code": "ValidationException", "Message": "not found"}},
            "DeleteEndpoint",
        )
        sm.delete_endpoint.side_effect = not_found
        sm.delete_endpoint_config.side_effect = not_found
        sm.delete_model.side_effect = not_found

        # Should not raise
        delete_endpoint("missing-ep", sm)

    def test_reraises_unexpected_error(self) -> None:
        sm = MagicMock()
        sm.delete_endpoint.side_effect = ClientError(
            {"Error": {"Code": "InternalError", "Message": "oops"}},
            "DeleteEndpoint",
        )
        with pytest.raises(ClientError):
            delete_endpoint("my-ep", sm)


# --- resolve_image_uri ---


class TestResolveImageUri:
    @patch("sagemaker.core.image_uris.retrieve")
    def test_returns_resolved_uri(self, mock_retrieve: MagicMock) -> None:
        mock_retrieve.return_value = "763104351884.dkr.ecr.us-east-1.amazonaws.com/hf:tag"
        result = resolve_image_uri("us-east-1", "ml.g5.xlarge")
        assert result == "763104351884.dkr.ecr.us-east-1.amazonaws.com/hf:tag"
        mock_retrieve.assert_called_once()

    @patch("sagemaker.core.image_uris.retrieve")
    def test_exits_on_value_error(self, mock_retrieve: MagicMock) -> None:
        mock_retrieve.side_effect = ValueError("Unsupported version")
        with pytest.raises(SystemExit):
            resolve_image_uri("us-east-1", "ml.g5.xlarge")


# --- deploy_model ---


class TestDeployModel:
    @patch("deploy.resolve_image_uri", return_value="763104351884.dkr.ecr.us-east-1/hf:tag")
    def test_deploy_creates_resources(self, mock_resolve: MagicMock) -> None:
        sm = MagicMock()
        waiter = MagicMock()
        sm.get_waiter.return_value = waiter

        result = deploy_model(
            model_data="s3://b/m.tar.gz",
            role_arn="arn:role",
            instance_type="ml.g5.xlarge",
            endpoint_name="test-ep",
            region="us-east-1",
            hf_token=None,
            image_uri=None,
            sm_client=sm,
        )

        assert result == "test-ep"
        sm.create_model.assert_called_once()
        model_kwargs = sm.create_model.call_args[1]
        assert model_kwargs["ModelName"] == "test-ep"
        assert model_kwargs["PrimaryContainer"]["ModelDataUrl"] == "s3://b/m.tar.gz"
        assert model_kwargs["ExecutionRoleArn"] == "arn:role"

        sm.create_endpoint_config.assert_called_once()
        sm.create_endpoint.assert_called_once()
        waiter.wait.assert_called_once()
        mock_resolve.assert_called_once_with("us-east-1", "ml.g5.xlarge")

    def test_deploy_with_custom_image(self) -> None:
        sm = MagicMock()
        waiter = MagicMock()
        sm.get_waiter.return_value = waiter

        deploy_model(
            model_data="s3://b/m.tar.gz",
            role_arn="arn:role",
            instance_type="ml.g5.xlarge",
            endpoint_name="test-ep",
            region="us-east-1",
            hf_token=None,
            image_uri="custom-image:tag",
            sm_client=sm,
        )

        container = sm.create_model.call_args[1]["PrimaryContainer"]
        assert container["Image"] == "custom-image:tag"

    @patch("deploy.resolve_image_uri", return_value="img:tag")
    def test_deploy_with_hf_token(self, _mock: MagicMock) -> None:
        sm = MagicMock()
        waiter = MagicMock()
        sm.get_waiter.return_value = waiter

        deploy_model(
            model_data="s3://b/m.tar.gz",
            role_arn="arn:role",
            instance_type="ml.g5.xlarge",
            endpoint_name="test-ep",
            region="us-east-1",
            hf_token="hf_abc",
            image_uri=None,
            sm_client=sm,
        )

        env = sm.create_model.call_args[1]["PrimaryContainer"]["Environment"]
        assert env["HUGGING_FACE_HUB_TOKEN"] == "hf_abc"

    @patch("deploy.resolve_image_uri", return_value="img:tag")
    def test_deploy_waiter_failure_exits(self, _mock: MagicMock) -> None:
        sm = MagicMock()
        waiter = MagicMock()
        sm.get_waiter.return_value = waiter
        waiter.wait.side_effect = Exception("Waiter timeout")

        with pytest.raises(SystemExit):
            deploy_model(
                model_data="s3://b/m.tar.gz",
                role_arn="arn:role",
                instance_type="ml.g5.xlarge",
                endpoint_name="test-ep",
                region="us-east-1",
                hf_token=None,
                image_uri=None,
                sm_client=sm,
            )


# --- main (integration-style with mocks) ---


class TestMain:
    @patch("deploy.boto3.Session")
    def test_dry_run_prints_config(
        self, mock_session_cls: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_session.region_name = "us-east-1"
        mock_sts = MagicMock()
        mock_session.client.return_value = mock_sts
        mock_sts.get_caller_identity.return_value = {"Account": "123456789"}

        main([
            "--model-data", "s3://bucket/model.tar.gz",
            "--role-arn", "arn:aws:iam::123:role/SageMaker",
            "--endpoint-name", "my-endpoint",
            "--dry-run",
        ])

        out = capsys.readouterr().out
        assert "Dry-run config" in out
        assert "s3://bucket/model.tar.gz" in out
        assert "arn:aws:iam::123:role/SageMaker" in out
        assert "us-east-1" in out
        assert "ml.g5.xlarge" in out
        assert "my-endpoint" in out

    @patch("deploy.boto3.Session")
    def test_delete_flow(
        self, mock_session_cls: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_session.region_name = "us-west-2"
        mock_sts = MagicMock()
        mock_sm = MagicMock()
        mock_session.client.side_effect = lambda svc: (
            mock_sts if svc == "sts" else mock_sm
        )
        mock_sts.get_caller_identity.return_value = {"Account": "123456789"}

        main([
            "--model-data", "s3://bucket/model.tar.gz",
            "--delete",
            "--endpoint-name", "test-ep",
        ])

        out = capsys.readouterr().out
        assert "Teardown complete" in out
        mock_sm.delete_endpoint.assert_called_once_with(EndpointName="test-ep")

    @patch("deploy.boto3.Session")
    def test_no_region_exits(self, mock_session_cls: MagicMock) -> None:
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_session.region_name = None

        with pytest.raises(SystemExit):
            main([
                "--model-data", "s3://bucket/model.tar.gz",
                "--endpoint-name", "test-ep",
            ])

    @patch("deploy.boto3.Session")
    def test_no_credentials_exits(self, mock_session_cls: MagicMock) -> None:
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_session.region_name = "us-east-1"
        mock_sts = MagicMock()
        mock_session.client.return_value = mock_sts
        mock_sts.get_caller_identity.side_effect = NoCredentialsError()

        with pytest.raises(SystemExit):
            main([
                "--model-data", "s3://bucket/model.tar.gz",
                "--endpoint-name", "test-ep",
            ])

    @patch("deploy.boto3.Session")
    def test_endpoint_already_exists_exits(
        self, mock_session_cls: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SAGEMAKER_ROLE_ARN", "arn:aws:iam::123:role/R")
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_session.region_name = "us-east-1"

        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123"}
        mock_s3 = MagicMock()
        mock_sm = MagicMock()
        # describe_endpoint succeeds => endpoint exists
        mock_sm.describe_endpoint.return_value = {"EndpointStatus": "InService"}

        def client_factory(svc: str) -> MagicMock:
            return {"sts": mock_sts, "s3": mock_s3, "sagemaker": mock_sm}[svc]

        mock_session.client.side_effect = client_factory

        with pytest.raises(SystemExit):
            main([
                "--model-data", "s3://bucket/model.tar.gz",
                "--endpoint-name", "test-ep",
            ])


# --- async inference ---


class TestAsyncDeploy:
    def test_async_flag_parsed(self) -> None:
        args = parse_args([
            "--model-data", "s3://b/m.tar.gz",
            "--endpoint-name", "my-ep",
            "--async-inference",
            "--async-s3-output", "s3://bucket/async",
        ])
        assert args.async_inference is True
        assert args.async_s3_output == "s3://bucket/async"

    @patch("deploy.resolve_image_uri", return_value="img:tag")
    def test_async_produces_async_inference_config(self, _mock: MagicMock) -> None:
        sm = MagicMock()
        waiter = MagicMock()
        sm.get_waiter.return_value = waiter

        deploy_model(
            model_data="s3://b/m.tar.gz",
            role_arn="arn:role",
            instance_type="ml.g5.xlarge",
            endpoint_name="test-ep",
            region="us-east-1",
            hf_token=None,
            image_uri=None,
            sm_client=sm,
            async_inference=True,
            async_s3_output="s3://bucket/async",
        )

        config_kwargs = sm.create_endpoint_config.call_args[1]
        assert "AsyncInferenceConfig" in config_kwargs
        async_cfg = config_kwargs["AsyncInferenceConfig"]
        assert async_cfg["OutputConfig"]["S3OutputPath"] == "s3://bucket/async/output/"
        assert async_cfg["OutputConfig"]["S3FailurePath"] == "s3://bucket/async/failure/"
        assert async_cfg["ClientConfig"]["MaxConcurrentInvocationsPerInstance"] == 1

    @patch("deploy.resolve_image_uri", return_value="img:tag")
    def test_async_without_s3_output_raises(self, _mock: MagicMock) -> None:
        sm = MagicMock()
        waiter = MagicMock()
        sm.get_waiter.return_value = waiter

        with pytest.raises(ValueError, match="async-s3-output"):
            deploy_model(
                model_data="s3://b/m.tar.gz",
                role_arn="arn:role",
                instance_type="ml.g5.xlarge",
                endpoint_name="test-ep",
                region="us-east-1",
                hf_token=None,
                image_uri=None,
                sm_client=sm,
                async_inference=True,
                async_s3_output=None,
            )

    @patch("deploy.resolve_image_uri", return_value="img:tag")
    def test_sync_deploy_has_no_async_config(self, _mock: MagicMock) -> None:
        sm = MagicMock()
        waiter = MagicMock()
        sm.get_waiter.return_value = waiter

        deploy_model(
            model_data="s3://b/m.tar.gz",
            role_arn="arn:role",
            instance_type="ml.g5.xlarge",
            endpoint_name="test-ep",
            region="us-east-1",
            hf_token=None,
            image_uri=None,
            sm_client=sm,
            async_inference=False,
        )

        config_kwargs = sm.create_endpoint_config.call_args[1]
        assert "AsyncInferenceConfig" not in config_kwargs

    @patch("deploy.boto3.Session")
    def test_async_dry_run(
        self, mock_session_cls: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_session.region_name = "us-east-1"
        mock_sts = MagicMock()
        mock_session.client.return_value = mock_sts
        mock_sts.get_caller_identity.return_value = {"Account": "123456789"}

        main([
            "--model-data", "s3://bucket/model.tar.gz",
            "--role-arn", "arn:aws:iam::123:role/SageMaker",
            "--endpoint-name", "my-endpoint",
            "--async-inference",
            "--async-s3-output", "s3://bucket/async",
            "--dry-run",
        ])

        out = capsys.readouterr().out
        assert "async" in out.lower()
        assert "s3://bucket/async" in out
