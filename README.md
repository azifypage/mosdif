# mosdif - Video Head Replacement

Replace full heads (face + hair + skull) in video inputs using a source portrait photo, powered by **LTX-2.5 Diffusion** (`Stubelius Remix v1` / `Lightricks LTX-2.5`).

## Core Features
- **2 Inputs**: Target Video (`input.mp4`) + Source Head Photo (`head.jpg`).
- **Full Head Replacement**: Replaces the entire head (face, facial features, hair volume, and skull contour) instead of a simple face-swap.
- **MediaPipe / OpenCV Landmark Alignment**: Automatically tracks the head in the video, smooths motion trajectory, aligns angle/scale/tilt, and harmonizes skin tones.
- **LTX-2.5 Neural Diffusion Refiner**: Renders seamless skin-to-neck transitions, realistic skin textures, hair physics, and scene lighting coherence.
- **Google Colab Ready**: Interactive notebook with `#@param` GUI controls, inline Base64 video previews, and high-speed `aria2c` downloads.

## Usage in Google Colab
Open `MosaicDiff_Colab.ipynb` in Google Colab (select GPU runtime: T4 / L4 / A100):
1. Run Cell 1 to 3 to prepare the workspace and dependencies (`mediapipe`, `ComfyUI`).
2. Run Cell 4 to download LTX-2.5 Stubelius Remix weights.
3. Configure Cell 5 with your video and head photo paths.
4. Run Cell 6 to execute Video Head Replacement.
5. Preview and download the result in Cell 7!

## CLI Usage
```bash
python cli.py video.mp4 -i head.jpg -o output/result.mp4 --denoise 0.45 --steps 8
```
