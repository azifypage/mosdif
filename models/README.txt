Public weights are downloaded into this folder on Start when they are missing.
LoRA is optional because Eros10 NSFW LoRA and DMD Distilled LoRA are already baked into the LTX-2.5 checkpoint.

basicvsr.pth                               BasicVSR++ checkpoint (plain weights, not a compiled engine)
rfdetr.onnx                                mosaic detector model (runs as-is on any graphics card)
ltx25_uncensored_v1.1-Q4_K_M.gguf          LTX-2.5 Uncensored v1.1 DiT (ChrisColeTech/LTX-2.5-uncensored-v1.1-FP8)
gemma4_12b_ltx25_uncensored-int8.safetensors Gemma-4 12B Text Encoder
ltx25_uncensored_video_vae.safetensors     LTX-2.5 Video VAE
lora.safetensors                           (Optional) Custom LoRA if needed

ComfyUI is required for the LTX-2.5 pass (along with the ComfyUI-GGUF custom node for GGUF weights).
Point ComfyUI Python and the ComfyUI folder at that install from the Models button or settings.

Licenses for the detector, BasicVSR++, and the bundled libraries are in
the licenses folder next to the program. Start with licenses\NOTICES.txt.
