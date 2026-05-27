# Workflow Examples

Load the UI workflow JSON files directly in ComfyUI. The API files are for
programmatic `/prompt` calls and will not display as a normal canvas workflow.

## Main Workflow

- `workflows/pngtuber_video_upload_bundle.json`

Upload or drag a video onto `PNGTuber Video Upload to Mouth Bundle`, then queue
the workflow. This is the default path for most users.

## Independent Post-Processing

- `workflows/pngtuber_generated_mouth_applier.json`

Use this after a first-pass bundle already exists. Set `bundle_manifest_path`
and provide one or more generated mouth images. The workflow updates the bundle
atlas without reprocessing the source video.

## Closed-Mouth Fallback

- `workflows/pngtuber_qwen_mouth_generation.json`

Use this only when the first pass marks the bundle with
`articulation.requiresModelGeneration: true`. It expects local Qwen Image Edit
nodes and models to be installed in ComfyUI.

## API Payloads

- `workflows/pngtuber_video_upload_bundle_api.json`
- `workflows/pngtuber_generated_mouth_applier_api.json`
- `workflows/pngtuber_qwen_mouth_generation_api.json`

These are plain ComfyUI API prompts for scripts and CI checks.
