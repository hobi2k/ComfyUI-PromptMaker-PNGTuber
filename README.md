# ComfyUI-PromptMaker-PNGTuber

Generic ComfyUI custom node for turning a character video into a video-mouth
PNGTuber asset bundle.

The package name keeps the PromptMaker integration target, but the pipeline is
not PromptMaker-only. It writes a general `pngtuber.videoMouthBundle.v1`
manifest, a frame-by-frame mouth track, flat compatibility sprites, and an
angle-aware mouth sprite atlas.

The node is local-only. It does not call OpenRouter or any remote image API.

## Nodes

- Class type: `PNGTuberVideoMouthBuilder`
- Class type: `PNGTuberVideoUploadToMouthBundle`
- Compatibility alias: `PromptMakerPNGTuberVideoMouth`
- Class type: `PNGTuberGeneratedMouthSpriteApplier`
- Category: `PromptMaker/PNGTuber`

`PNGTuberVideoUploadToMouthBundle` is the preferred one-node workflow. It shows
a ComfyUI input-folder video selector plus an `upload video` button and can also
accept drag-and-drop video files. Run it by selecting/uploading one video and
queueing the workflow.

`PNGTuberVideoMouthBuilder` is the advanced compatibility node. It exposes the
same extraction pipeline but keeps the lower-level `video_path` input and tuning
controls for scripts or older workflows.

`PNGTuberGeneratedMouthSpriteApplier` takes locally generated Qwen Image Edit
outputs and replaces or fills `closed/open/half/e/u` sprites inside an existing
bundle. Use it when the builder marks a closed-mouth or weak-articulation video
with `articulation.requiresModelGeneration: true`.

## Outputs

For each input video, the node writes:

- `loop_mouthless_h264.mp4`
- `mouth_track.json`
- `mouth_sprite_atlas.json`
- `bundle_manifest.json`
- `summary.json`
- `mouth/closed.png`
- `mouth/half.png`
- `mouth/open.png`
- `mouth/e.png`
- `mouth/u.png`
- `mouth/angles/angle_m45/{closed,half,open,e,u}.png`
- `mouth/angles/angle_m30/{closed,half,open,e,u}.png`
- `mouth/angles/angle_m15/{closed,half,open,e,u}.png`
- `mouth/angles/angle_p00/{closed,half,open,e,u}.png`
- `mouth/angles/angle_p15/{closed,half,open,e,u}.png`
- `mouth/angles/angle_p30/{closed,half,open,e,u}.png`
- `mouth/angles/angle_p45/{closed,half,open,e,u}.png`

## Schemas

- `bundle_manifest.json`: `pngtuber.videoMouthBundle.v1`
- `mouth_track.json`: `pngtuber.mouthTrack.v1`
- `mouth_sprite_atlas.json`: `pngtuber.mouthSpriteAtlas.v1`

`mouth_track.json` includes:

- `frames[].quad`: four-point mouth placement quad in source video coordinates
- `frames[].mouthOpen`: normalized mouth openness estimate
- `frames[].mouthAngleDegrees`: estimated mouth angle
- `frames[].spriteSet`: nearest atlas set such as `angle_m15` or `angle_p00`
- `frames[].qualityScore`: mouth signal quality used for sprite candidate selection
- `frames[].occlusionScore`: high values indicate likely occlusion or weak mouth signal

If the source video does not contain enough mouth-open variation, the bundle is
marked with `articulation.requiresModelGeneration: true` and the node writes:

- `mouth_generation_inputs/reference_face_mouth_crop.png`
- `mouth_generation_inputs/mouth_edit_mask.png`
- `mouth_generation_inputs/mouth_generation_plan.json`

That plan is for a local image-edit model pass, not a remote API call. It is
designed for installed ComfyUI Qwen Image Edit assets such as
`qwen_image_edit_2511_bf16_각도.safetensors` plus the local Qwen text encoder and
VAE. In this case, extracted mouth sprites should be treated as provisional
until `PNGTuberGeneratedMouthSpriteApplier` replaces them with model-generated
sprites.

The applier accepts optional `closed_image`, `open_image`, `half_image`,
`e_image`, and `u_image` ComfyUI `IMAGE` inputs. It extracts the mouth component
from those edited face crops, writes transparent mouth sprites, rotates them
across the atlas angle bins, and updates `bundle_manifest.json`,
`mouth_sprite_atlas.json`, and `summary.json` with `modelGeneration` metadata.

The flat five mouth sprites keep compatibility with players that only support a
single mouth set. The angle atlas is for smoother runtime animation when a
player supports `mouth_sprite_atlas.json` and `frames[].spriteSet`.

## Install

Clone into ComfyUI's `custom_nodes` directory:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/hobi2k/ComfyUI-PromptMaker-PNGTuber.git
```

Install dependencies with the Python environment used by ComfyUI:

```bash
cd /path/to/ComfyUI
python -m pip install -r custom_nodes/ComfyUI-PromptMaker-PNGTuber/requirements.txt
```

Restart ComfyUI after installing the node or dependencies.

## Detection Pipeline

Default mode is `anime_first`.

1. Detect anime faces with `models/lbpcascade_animeface.xml`.
2. Refine the mouth region with local OpenCV edge/dark/lip-color analysis.
3. Fall back to MediaPipe FaceMesh if anime detection fails.
4. Fall back to a local ComfyUI YOLO face model when available at
   `models/ultralytics/bbox/face_yolov8m.pt`.
5. Interpolate missing frames and smooth the mouth track.
6. Inpaint the mouth region from every video frame using OpenCV TELEA.
7. Score each frame for mouth signal quality and likely occlusion.
8. Measure mouth-open variation. Closed-only or near-closed-only videos are
   flagged for local model generation instead of trusting forced extraction.
9. Extract mouth-only transparent sprites from the best non-occluded candidates.
10. Build angle-aware mouth sprite sets. The default range is `-45` through
   `+45` degrees in 15 degree steps, configurable by `angle_range_degrees` and
   `angle_step_degrees`. Missing bins are generated from the nearest real
   sprite set and are marked with `generated: true` in
   `mouth_sprite_atlas.json`.
11. Write atlas-level QA metadata under `quality`, including real/generated
   angle sets, frame quality, occlusion ratio, and warnings.

For closed-mouth videos, run a local Qwen Image Edit workflow with
`mouth_generation_inputs/reference_face_mouth_crop.png` as the reference image,
then connect the generated shape images to `PNGTuberGeneratedMouthSpriteApplier`.
The applier can replace all five shapes, including `closed`, so a poor forced
extraction is not kept just because the video had no usable articulation.

## Workflow Example

UI-loadable workflows are included at:

```text
examples/workflows/promptmaker_pngtuber_video_upload_bundle.json
examples/workflows/promptmaker_pngtuber_video_mouth.json
examples/workflows/promptmaker_pngtuber_generated_mouth_applier.json
examples/workflows/promptmaker_pngtuber_qwen_mouth_generation.json
```

These files use ComfyUI's canvas/LiteGraph format and show nodes when loaded
from the ComfyUI workflow menu or by drag-and-drop.

For a normal video-to-PNGTuber bundle run, open
`promptmaker_pngtuber_video_upload_bundle.json`, press `upload video` on the
single `PNGTuber Video Upload to Mouth Bundle` node, select a local video, and
queue the prompt. The node writes the complete PromptMaker-ready bundle under
ComfyUI output. `promptmaker_pngtuber_video_mouth.json` points at the same
one-node upload workflow for compatibility with older documentation.

API `/prompt` examples are included at:

```text
examples/workflows/promptmaker_pngtuber_video_mouth_api.json
examples/workflows/promptmaker_pngtuber_generated_mouth_applier_api.json
examples/workflows/promptmaker_pngtuber_qwen_mouth_generation_api.json
```

`promptmaker_pngtuber_qwen_mouth_generation_api.json` is the local-only
closed-mouth fallback pass: it loads `reference_face_mouth_crop.png`, generates
`closed/open/half/e/u` with Qwen Image Edit, then feeds those decoded images
directly into `PNGTuberGeneratedMouthSpriteApplier`.

Restart ComfyUI after updating the custom node so the applier class is visible
in the node registry.

If a workflow shows red `LoadImage` nodes, upload or replace the placeholder
input file names with the actual generated mouth images in ComfyUI's input
folder. Red missing-input nodes still mean the workflow loaded; a blank canvas
means an API JSON was loaded into the UI by mistake.

Use the upload button or drag a video onto the node. Absolute paths are still
supported through the advanced node or the `advanced_video_path` override, but
they are no longer the normal UI path.

## PromptMaker Compatibility

PromptMaker `pngtuber_mode=video_mouth` accepts both the flat compatibility
files and the angle-aware atlas bundle:

- `loop_mouthless_h264.mp4`
- `mouth_track.json`
- `mouth_sprite_atlas.json`
- `mouth/closed.png`
- `mouth/half.png`
- `mouth/open.png`
- optional `mouth/e.png`
- optional `mouth/u.png`
- optional `mouth/angles/*/{closed,half,open,e,u}.png`

PromptMaker uses `frames[].spriteSet` first and falls back to
`frames[].mouthAngleDegrees` when selecting an angle sprite.

## License Notes

`lbpcascade_animeface.xml` is included from `nagadomi/lbpcascade_animeface`
under the MIT license. Its license header is preserved inside the XML file.
