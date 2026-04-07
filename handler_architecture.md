# Current workflow

1. package_model.py — Downloads HF weights, copies inference.py into code/, bundles everything into
  model.tar.gz, which you upload to S3.
  2. deploy.py — Creates a SageMaker endpoint using the HF DLC (Deep Learning Container) image. The container   
  comes pre-installed with torch, transformers, accelerate, etc. SageMaker pulls your model.tar.gz from S3,     
  extracts it, and the container automatically discovers code/inference.py.
  3. inference.py — The container calls its four functions as the SageMaker Inference Toolkit contract:
    - model_fn → loads the model/processor from the extracted weights directory
    - input_fn → deserializes the incoming JSON request
    - predict_fn → runs the actual inference
    - output_fn → serializes the response back to JSON
  4. Client-side — You invoke the endpoint via boto3 (sagemaker-runtime.invoke_endpoint), sending JSON with text
   and optional base64 image. SageMaker routes it through the four functions above and returns the response.    

  The key thing: inference.py runs inside the container on the SageMaker instance, not locally. Your local code 
  just packages it and deploys it. The container handles the HTTP serving layer — your code only needs to
  implement those four functions.





# SageMaker Handler Architecture — Design Document

## Context

The current repo has a single MedGemma deployment with hardcoded model-specific logic throughout `inference.py`, `deploy.py`, and `package_model.py`. To support additional models (both HuggingFace and non-HF like TotalSegmentator), we need an architecture that separates model-specific config from shared infrastructure.

This is a **design document only** — no implementation yet.

---

## 1. Abstract Base Handler

All handlers implement SageMaker's four-method contract:

```
src/handlers/base.py — BaseHandler(ABC)

  model_fn(model_dir)        → Load model artifacts into memory
  input_fn(body, content_type) → Deserialize request into internal format
  predict_fn(input_data, model) → Run inference
  output_fn(prediction, accept) → Serialize response
```

Each model's `model.tar.gz` ships a thin `code/inference.py` shim that delegates to the handler:

```python
from handlers.hf_handler import HFHandler
_handler = HFHandler("model_config.json")

def model_fn(model_dir):      return _handler.model_fn(model_dir)
def input_fn(body, ct):       return _handler.input_fn(body, ct)
def predict_fn(data, model):  return _handler.predict_fn(data, model)
def output_fn(pred, accept):  return _handler.output_fn(pred, accept)
```

The base class is a pure ABC — shared utilities (base64 decoding, JSON serialization) live in `handlers/utils.py`, not in the base class.

---

## 2. HF Handler (Config-Driven)

A single `HFHandler` class serves **all** HuggingFace transformers models. A JSON config per model parameterizes behavior. Adding a new HF model = adding a new JSON file, no code changes.

Axes of variation it handles:
- Auto class (`AutoModelForImageTextToText`, `AutoModelForCausalLM`, etc.)
- Processor class (`AutoProcessor`, `AutoTokenizer`, `AutoFeatureExtractor`)
- dtype (`bfloat16`, `float16`, `float32`)
- Input modality (`text`, `image`, `text+image`)
- Chat template usage (on/off)
- Generation params (`max_new_tokens`, `do_sample`, `temperature`)
- Output key name (`generated_text`, `classification`, `similarity_scores`)

---

## 3. Model Config Schema (HF Models)

Each HF model ships a `model_config.json` at the root of its `model.tar.gz`:

```json
{
  "model_id": "google/medgemma-1.5-4b-it",
  "display_name": "MedGemma 4B",
  "auto_class": "AutoModelForImageTextToText",
  "processor_class": "AutoProcessor",
  "dtype": "bfloat16",
  "device_map": "auto",
  "input_modality": "text+image",
  "use_chat_template": true,
  "generation_params": {
    "max_new_tokens": 512,
    "do_sample": false
  },
  "output_key": "generated_text",
  "requires_hf_token": true
}
```

| Field | Required | Default | Purpose |
|---|---|---|---|
| `model_id` | yes | — | HF model identifier (provenance, not loading) |
| `auto_class` | yes | — | transformers Auto class name |
| `processor_class` | no | `AutoProcessor` | Processor/tokenizer class |
| `dtype` | no | `bfloat16` | Torch dtype |
| `device_map` | no | `auto` | Device placement |
| `input_modality` | yes | — | `text`, `image`, or `text+image` |
| `use_chat_template` | no | `true` | Chat vs direct tokenization |
| `generation_params` | no | `{}` | kwargs for `model.generate()` |
| `output_key` | no | `generated_text` | Response JSON key |
| `requires_hf_token` | no | `false` | Needs HF_TOKEN in container env |

---

## 4. Custom Handler Pattern (TotalSegmentator)

Non-HF models get their own handler subclass. They share the base contract but have completely different implementations.

**Why TotalSegmentator can't use HFHandler:**
- nnU-Net weights, not `from_pretrained()`
- Binary I/O (NIfTI volumes), not JSON
- 3D patch-based sliding window inference
- Dependencies: `nnunetv2`, `nibabel`, `SimpleITK` (not in HF DLC)
- Custom Docker container required
- Minutes per inference, not seconds

```
src/handlers/totalsegmentator_handler.py — TotalSegmentatorHandler(BaseHandler)

  model_fn  → Load nnU-Net predictor with task-specific plans
  input_fn  → Deserialize NIfTI from binary bytes (or fetch from S3 URI)
  predict_fn → Patch-based 3D inference with sliding window + ensemble
  output_fn → Serialize segmentation mask to NIfTI bytes (or write to S3)
```

**Pattern**: Any non-HF model → new file in `src/handlers/`, subclass `BaseHandler`, implement the four methods.

---

## 5. Shared Infrastructure

### deploy.py — Mostly reusable already

| Function | Reusable? | Change needed |
|---|---|---|
| `deploy_model()` | Yes | Model-agnostic (container image + S3 URI + role) |
| `delete_endpoint()` | Yes | No changes |
| `validate_s3_object()` | Yes | No changes |
| `parse_args()` | Yes | Remove `medgemma-endpoint` default, make `--endpoint-name` required |
| `resolve_image_uri()` | HF only | Non-HF models pass `--image-uri` explicitly (already supported) |
| `get_hf_token()` | HF only | Rename `MED_GEM_TOKEN` → generic `HF_TOKEN` |

### package_model.py — Needs splitting

| Function | Reusable? | Notes |
|---|---|---|
| `create_archive()` | Yes | tar.gz creation is universal |
| `check_disk_space()` | Yes | Threshold should come from config |
| `stage_code()` | Partially | Which `inference.py` shim to copy is handler-specific |
| `download_model()` | HF only | Non-HF models have different artifact sources |
| `clean_staging()` | HF only | HF-specific cache cleanup |

---

## 6. Container Strategy

| Model Type | Container | How |
|---|---|---|
| HF models | AWS HuggingFace DLC | `resolve_image_uri()` (existing) |
| TotalSegmentator | Custom ECR image | `docker/totalsegmentator/Dockerfile` + `--image-uri` flag |

Custom images need: `sagemaker-inference` toolkit + model-specific deps. Handler code ships in `model.tar.gz/code/` (not baked into image) for flexibility.

---

## 7. I/O Contract

No single content type forced — each handler declares what it supports.

| Model Type | Input | Output |
|---|---|---|
| HF (text gen) | `application/json` — `{"text": "...", "image": "<base64>"}` | `application/json` — `{"generated_text": "..."}` |
| HF (classifier) | `application/json` — `{"image": "<base64>"}` | `application/json` — `{"classification": [...]}` |
| TotalSegmentator | `application/json` — `{"s3_uri": "s3://..."}` | `application/json` — `{"output_s3_uri": "s3://..."}` |

**Large payloads** (3D volumes): Use S3 URI pattern instead of inline binary. SageMaker `invoke_endpoint` has a 6 MB payload limit. Handler fetches from S3, writes result back, returns URI.

---

## 8. Proposed Directory Structure

```
src/
  handlers/
    __init__.py
    base.py                        # BaseHandler ABC
    hf_handler.py                  # Config-driven HF handler
    totalsegmentator_handler.py    # Custom TS handler
    utils.py                       # Shared utilities
  configs/
    medgemma.json
    biomedclip.json
  shims/
    hf_inference.py                # SageMaker shim for HF models
    totalseg_inference.py          # SageMaker shim for TotalSeg
  deploy.py                        # Generalized (no model-specific defaults)
  package_model.py                 # Refactored with pluggable packagers
docker/
  totalsegmentator/
    Dockerfile
```

---

## 9. Open Questions

1. **Handler distribution** — Ship handler classes in `model.tar.gz/code/` (flexible, duplicated) or publish as a pip package in the container (DRY, requires container rebuild to update)?

2. **Encoder-only models** — `HFHandler.predict_fn` currently assumes `model.generate()`. Encoder models (BiomedCLIP) use `model.forward()`. Add `inference_mode` config field (`generate` vs `forward` vs `encode`)? Or subclass into `HFGenerativeHandler` / `HFEncoderHandler`?

3. **Async inference for TotalSegmentator** — Real-time endpoints have a 60s timeout (extendable to 15 min). 3D segmentation can take minutes. SageMaker Async Inference may be more appropriate. Investigate before implementing.

4. **Config validation** — Use `jsonschema` or Pydantic to validate `model_config.json` at load time? Prevents cryptic runtime errors but adds a dependency.

5. **Inference Components** — `usr.md` documents interest in multi-model endpoints. The handler architecture is endpoint-agnostic, but `deploy.py` would need `create_inference_component` support.
