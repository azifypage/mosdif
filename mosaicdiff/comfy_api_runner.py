"""ComfyUI Headless API Runner for BFS Head Swap Workflow.

Executes the official workflow_ltx2.5_v1.1_head_swap_drag_and_drop.json directly
via ComfyUI's REST & WebSocket API without requiring a web browser interface.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path


class ComfyAPIError(RuntimeError):
    pass


REQUIRED_BFS_MODELS = [
    (
        "diffusion_models",
        "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
        "https://huggingface.co/osantinello/LTX25_Models/resolve/main/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
    ),
    (
        "loras",
        "head_swap_ltx25_r128_v1.1.safetensors",
        "https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap-Video/resolve/main/ltx-2.5/head_swap_ltx25_r128_v1.1.safetensors",
    ),
    (
        "text_encoders",
        "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
        "https://huggingface.co/osantinello/LTX25_Models/resolve/main/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
    ),
    (
        "vae",
        "ltx-2.5-video-vae-bf16.safetensors",
        "https://huggingface.co/osantinello/LTX25_Models/resolve/main/ltx-2.5-video-vae-bf16.safetensors",
    ),
    (
        "vae",
        "ltx-2.5-audio-vae-bf16.safetensors",
        "https://huggingface.co/osantinello/LTX25_Models/resolve/main/ltx-2.5-audio-vae-bf16.safetensors",
    ),
]

SECOND_PASS_MODELS = [
    (
        "latent_upscale_models",
        "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        "https://huggingface.co/osantinello/LTX25_Models/resolve/main/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    ),
]


def is_comfy_running(host: str = "127.0.0.1", port: int = 8188) -> bool:
    """Check if ComfyUI server is actively responding on port 8188."""
    url = f"http://{host}:{port}/system_stats"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MosaicDiff-Runner"})
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def stop_comfy_daemon(
    host: str = "127.0.0.1",
    port: int = 8188,
    log_cb=print,
) -> None:
    """Stop running ComfyUI daemon so it can refresh and re-index models."""
    if not is_comfy_running(host, port):
        return
    log_cb("Menghentikan ComfyUI daemon untuk re-indexing model...")
    if sys.platform != "win32":
        try:
            subprocess.run(f"fuser -k -9 {port}/tcp 2>/dev/null || true", shell=True)
            subprocess.run(["pkill", "-9", "-f", "main.py"], stderr=subprocess.DEVNULL)
        except Exception:
            pass
    time.sleep(2.5)


def is_model_indexed_by_comfy(
    class_type: str,
    input_name: str,
    model_name: str,
    host: str = "127.0.0.1",
    port: int = 8188,
) -> bool:
    """Check if ComfyUI's object_info API lists model_name for a given node class."""
    url = f"http://{host}:{port}/object_info/{class_type}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MosaicDiff-Runner"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        inputs = data.get(class_type, {}).get("input", {}).get("required", {})
        choices = inputs.get(input_name, [[]])[0]
        return model_name in choices
    except Exception:
        return False


def ensure_bfs_models(
    comfy_root: str | Path = "/content/ComfyUI",
    enable_second_pass: bool = False,
    log_cb=print,
) -> bool:
    """Verify and auto-download any missing official BFS model weights.
    
    Returns True if any model was newly downloaded or linked, indicating that
    the ComfyUI daemon should be restarted to re-index its model catalog.
    """
    c_root = Path(comfy_root).resolve()
    models_dir = c_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    targets = list(REQUIRED_BFS_MODELS)
    if enable_second_pass:
        targets.extend(SECOND_PASS_MODELS)

    # Alternate local locations where weights might already reside
    search_roots = [
        Path("/content/mosdif/models"),
        Path(__file__).resolve().parent.parent / "models",
        Path.home() / ".cache" / "huggingface" / "hub",
    ]

    downloaded_any = False

    for subf, fname, url in targets:
        dest_dir = models_dir / subf
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / fname

        # 1. Check if already present and non-empty (> 1MB)
        if dest_file.is_file() and dest_file.stat().st_size > 1024 * 1024:
            continue

        # 2. Check alternate local directories
        found_local = False
        for s_root in search_roots:
            cand = s_root / subf / fname
            if not cand.is_file():
                cand = s_root / fname
            if cand.is_file() and cand.stat().st_size > 1024 * 1024:
                log_cb(f"Linking local model {fname} -> models/{subf}/...")
                try:
                    if dest_file.is_symlink() or dest_file.is_file():
                        dest_file.unlink()
                    dest_file.symlink_to(cand)
                    found_local = True
                    downloaded_any = True
                    break
                except Exception:
                    shutil.copy2(cand, dest_file)
                    found_local = True
                    downloaded_any = True
                    break

        if found_local:
            continue

        # 3. Download directly from high-speed public repository
        log_cb(f"⚡ Mengunduh model pendukung ({fname})...")
        downloaded_any = True

        aria2_bin = shutil.which("aria2c")
        if aria2_bin:
            cmd = [
                aria2_bin,
                "-c",
                "-x", "16",
                "-s", "16",
                "-k", "1M",
                "-d", str(dest_dir),
                "-o", fname,
                url,
            ]
            res = subprocess.run(cmd)
            if res.returncode != 0:
                raise ComfyAPIError(f"Gagal mengunduh {fname} via aria2c (exit code {res.returncode})")
        else:
            req = urllib.request.Request(url, headers={"User-Agent": "MosaicDiff-Downloader"})
            with urllib.request.urlopen(req) as resp, open(dest_file, "wb") as out_f:
                shutil.copyfileobj(resp, out_f)

        if not dest_file.is_file() or dest_file.stat().st_size < 1024 * 1024:
            raise ComfyAPIError(f"Model {fname} gagal diunduh atau ukuran file korup.")

        size_mb = dest_file.stat().st_size / (1024 * 1024)
        log_cb(f"[OK] Model siap: {fname} ({size_mb:.1f} MB)")

    return downloaded_any


def start_comfy_daemon(
    comfy_root: str | Path = "/content/ComfyUI",
    python_bin: str = sys.executable,
    host: str = "127.0.0.1",
    port: int = 8188,
    log_cb=print,
    timeout_sec: int = 60,
    restart: bool = False,
) -> subprocess.Popen | None:
    """Ensure ComfyUI backend server is running in background."""
    if restart:
        stop_comfy_daemon(host, port, log_cb=log_cb)

    if is_comfy_running(host, port):
        log_cb(f"ComfyUI backend already running on http://{host}:{port}")
        return None

    comfy_dir = Path(comfy_root).resolve()
    main_py = comfy_dir / "main.py"
    if not main_py.is_file():
        raise ComfyAPIError(f"ComfyUI main.py tidak ditemukan di: {main_py}")

    log_cb(f"Starting headless ComfyUI daemon on http://{host}:{port}...")
    log_file = comfy_dir / "comfy_daemon.log"
    out_f = open(log_file, "a", encoding="utf-8")

    cmd = [
        str(python_bin),
        str(main_py),
        "--listen",
        host,
        "--port",
        str(port),
        "--disable-auto-launch",
    ]

    proc = subprocess.Popen(
        cmd,
        cwd=str(comfy_dir),
        stdout=out_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    # Wait until server responds
    start_time = time.time()
    while time.time() - start_time < timeout_sec:
        if is_comfy_running(host, port):
            log_cb(f"ComfyUI daemon is ready and responding on http://{host}:{port}!")
            return proc
        if proc.poll() is not None:
            out_f.close()
            err_log = log_file.read_text(encoding="utf-8", errors="replace")[-2000:]
            raise ComfyAPIError(f"ComfyUI daemon exited prematurely:\n{err_log}")
        time.sleep(1.5)

    out_f.close()
    raise ComfyAPIError(f"ComfyUI daemon failed to respond within {timeout_sec}s.")


def prepare_comfy_inputs(
    comfy_root: str | Path,
    video_input_path: str | Path,
    head_image_path: str | Path,
) -> tuple[str, str]:
    """Copy video and head image into ComfyUI/input/ folder and return filenames."""
    c_in = Path(comfy_root).resolve() / "input"
    c_in.mkdir(parents=True, exist_ok=True)

    v_src = Path(video_input_path).resolve()
    h_src = Path(head_image_path).resolve()

    if not v_src.is_file():
        raise FileNotFoundError(f"Input video file tidak ditemukan: {v_src}")
    if not h_src.is_file():
        raise FileNotFoundError(f"Input foto kepala tidak ditemukan: {h_src}")

    v_dst = c_in / v_src.name
    h_dst = c_in / h_src.name

    if not v_dst.is_file() or v_src.stat().st_mtime > v_dst.stat().st_mtime:
        shutil.copy2(v_src, v_dst)

    if not h_dst.is_file() or h_src.stat().st_mtime > h_dst.stat().st_mtime:
        shutil.copy2(h_src, h_dst)

    return v_src.name, h_src.name


OFFICIAL_BFS_API_WORKFLOW = (
    Path(__file__).resolve().parent.parent / "workflow_ltx2.5_v1.1_head_swap_drag_and_drop.api.json"
)


def load_bfs_prompt(
    api_path: str | Path | None = None,
    log_cb=print,
) -> dict:
    """Load the official BFS API prompt graph directly from disk.
    
    This is the primary deterministic execution path: it directly loads the
    validated 56-node graph without relying on third-party converter servers,
    fragile network calls, or guesswork fallbacks.
    """
    if api_path is None:
        target = OFFICIAL_BFS_API_WORKFLOW
        if not target.is_file():
            target = Path("/content/mosdif/workflow_ltx2.5_v1.1_head_swap_drag_and_drop.api.json")
    else:
        target = Path(api_path).resolve()
        if not target.name.endswith(".api.json"):
            companion = target.with_name(target.stem + ".api.json")
            if companion.is_file():
                target = companion

    if not target.is_file():
        raise FileNotFoundError(
            f"Official BFS API workflow file not found at: {target}\n"
            f"Pastikan repository mosdif ter-clone dengan lengkap."
        )

    with open(target, "r", encoding="utf-8") as f:
        prompt_dict = json.load(f)

    if not isinstance(prompt_dict, dict) or not any("class_type" in v for v in prompt_dict.values()):
        raise ComfyAPIError(f"Format file API prompt tidak valid: {target}")

    if log_cb:
        log_cb(f"[OK] Berhasil memuat workflow resmi BFS API: {target.name} ({len(prompt_dict)} nodes)")

    return prompt_dict


def convert_workflow_to_prompt(
    workflow_path: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8188,
    log_cb=print,
) -> dict:
    """Primary loader alias for backwards compatibility."""
    return load_bfs_prompt(api_path=workflow_path, log_cb=log_cb)


def patch_bfs_prompt(
    prompt_dict: dict,
    video_input_path: str | Path,
    head_image_path: str | Path,
    prompt_text: str = "",
    negative_prompt_text: str = "",
    lora_strength: float = 0.80,
    steps: int = 8,
    seed: int = 42,
    enable_second_pass: bool = False,
    comfy_root: str | Path = "/content/ComfyUI",
) -> dict:
    """Patch the API prompt graph nodes with user-provided parameters."""
    patched = json.loads(json.dumps(prompt_dict))  # deep copy

    v_filename, h_filename = prepare_comfy_inputs(comfy_root, video_input_path, head_image_path)

    # Format BFS Head Swap prompt
    p_text = prompt_text.strip()
    if not p_text.lower().startswith("head_swap:") and not p_text.lower().startswith("head swap"):
        if p_text:
            pos_prompt = f"head_swap:\nFACE:\n{p_text}"
        else:
            pos_prompt = "head_swap:\nFACE:\nPhotorealistic human face, natural skin texture, sharp detailed eyes, realistic lighting"
    else:
        pos_prompt = p_text

    neg_prompt = (
        negative_prompt_text.strip()
        or "mouth movement, cartoon, anime, illustration, 3d render, plastic, painting, blurry, deformed"
    )

    # Check available lora files in ComfyUI
    lora_dir = Path(comfy_root).resolve() / "models" / "loras"
    available_loras = [f.name for f in lora_dir.glob("*.safetensors")] if lora_dir.is_dir() else []

    nodes_to_remove = []

    for nid, node in patched.items():
        ctype = node.get("class_type", "")
        inputs = node.get("inputs", {})
        title = node.get("_meta", {}).get("title", "")

        # 1. Target Guide Video Input (VHS_LoadVideo, Node 460)
        if ctype == "VHS_LoadVideo" or "Guide Reference" in title:
            if "video" in inputs:
                inputs["video"] = v_filename

        # 2. Head Image Input (LoadImage, Node 461)
        if ctype == "LoadImage" or "Face Reference" in title:
            if "image" in inputs:
                inputs["image"] = h_filename

        # 3. LoRA Model Loader (LoraLoaderModelOnly, Node 610)
        if ctype in ("LoraLoaderModelOnly", "LoraLoader") and "head_swap" in str(inputs.get("lora_name", "")).lower():
            if "strength_model" in inputs:
                inputs["strength_model"] = float(lora_strength)
            # Use official canonical file or matching file on disk
            cur_lora = inputs.get("lora_name", "")
            if cur_lora not in available_loras:
                for cand in available_loras:
                    if "head_swap" in cand.lower():
                        inputs["lora_name"] = cand
                        break

        # 4. Positive Prompt (CLIPTextEncode, Node 448)
        if ctype == "CLIPTextEncode" and ("Positive" in title or nid == "448"):
            inputs["text"] = pos_prompt

        # 5. Negative Prompt (CLIPTextEncode, Node 444)
        if ctype == "CLIPTextEncode" and ("Negative" in title or nid == "444"):
            inputs["text"] = neg_prompt

        # 6. Samplers & Schedulers
        if "seed" in inputs or "noise_seed" in inputs:
            k = "noise_seed" if "noise_seed" in inputs else "seed"
            inputs[k] = int(seed)

        if ctype in ("BasicScheduler", "LTXVScheduler") and "steps" in inputs:
            inputs["steps"] = int(steps)

        # 7. UNETLoader (Node 443) required weight_dtype
        if ctype == "UNETLoader" or nid == "443":
            inputs["weight_dtype"] = "default"

        # 8. CLIPLoader (Node 446) required type
        if ctype == "CLIPLoader" or nid == "446":
            inputs.setdefault("type", "ltxv")

        # 9. Second Pass & Comparison Video Handling
        if not enable_second_pass:
            if ctype == "SaveVideo" and ("Second Pass" in title or nid == "75"):
                nodes_to_remove.append(nid)
            elif ctype == "VHS_VideoCombine" and ("comparison" in str(inputs).lower() or nid == "491"):
                nodes_to_remove.append(nid)

        # 10. Math Expression for video length (Node 455: a * b + 1)
        if nid == "455" or (ctype == "ComfyMathExpression" and "b" in str(inputs.get("expression", ""))):
            if "values.b" not in inputs and "b" not in inputs:
                inputs["values.b"] = ["430", 0]

    # Prune second pass save nodes if disabled so ComfyUI only generates first pass
    for nid in nodes_to_remove:
        patched.pop(nid, None)

    return patched


def run_comfy_workflow(
    prompt_dict: dict,
    dest_file: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8188,
    comfy_root: str | Path = "/content/ComfyUI",
    log_cb=print,
    progress_cb=None,
    cancel_event=None,
) -> Path:
    """Submit prompt to ComfyUI API, stream progress, and wait for generated output video."""
    # 0. Self-Healing Model Check: Pastikan seluruh file bobot BFS tersedia di ComfyUI/models/
    has_second_pass = any(
        n.get("class_type") == "SaveVideo" and "Second Pass" in n.get("_meta", {}).get("title", "")
        for n in prompt_dict.values()
    )
    needed_restart = ensure_bfs_models(
        comfy_root=comfy_root,
        enable_second_pass=has_second_pass,
        log_cb=log_cb,
    )

    # Verifikasi apakah ComfyUI daemon sudah mengindeks model video VAE
    if is_comfy_running(host, port):
        if needed_restart or not is_model_indexed_by_comfy("VAELoader", "vae_name", "ltx-2.5-video-vae-bf16.safetensors", host, port):
            log_cb("Sinkronisasi model ke ComfyUI daemon (restart untuk re-indexing)...")
            stop_comfy_daemon(host=host, port=port, log_cb=log_cb)
            start_comfy_daemon(comfy_root=comfy_root, host=host, port=port, log_cb=log_cb)
    else:
        start_comfy_daemon(comfy_root=comfy_root, host=host, port=port, log_cb=log_cb)

    client_id = str(uuid.uuid4())
    url = f"http://{host}:{port}/prompt"

    payload = {
        "prompt": prompt_dict,
        "client_id": client_id,
    }

    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=req_data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            res = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as http_err:
        err_body = http_err.read().decode("utf-8", errors="replace")
        try:
            err_json = json.loads(err_body)
            err_details = json.dumps(err_json, indent=2)
        except Exception:
            err_details = err_body
        log_cb(f"❌ ComfyUI /prompt menolak workflow (HTTP {http_err.code}):\n{err_details}")
        raise ComfyAPIError(f"ComfyUI /prompt rejected with HTTP {http_err.code}:\n{err_details}") from http_err

    prompt_id = res.get("prompt_id")
    if not prompt_id:
        raise ComfyAPIError(f"Gagal mengirim workflow ke ComfyUI: {res}")

    log_cb(f"Workflow berhasil dimasukkan ke antrean ComfyUI (Prompt ID: {prompt_id})")

    # Connect to WebSocket for real-time progress & node execution
    ws = None
    try:
        import websocket
        ws_url = f"ws://{host}:{port}/ws?clientId={client_id}"
        ws = websocket.create_connection(ws_url, timeout=10)
        log_cb("Terhubung ke WebSocket ComfyUI untuk pemantauan live.")
    except Exception as exc:
        log_cb(f"Notice: WebSocket tidak aktif ({exc}), fallback ke polling HTTP.")

    # Poll execution until done
    completed = False

    try:
        while not completed:
            if cancel_event and cancel_event.is_set():
                try:
                    c_req = urllib.request.Request(f"http://{host}:{port}/interrupt", data=b"{}")
                    urllib.request.urlopen(c_req)
                except Exception:
                    pass
                raise InterruptedError("Proses dibatalkan oleh pengguna.")

            if ws is not None:
                try:
                    msg = ws.recv()
                    if isinstance(msg, str):
                        data = json.loads(msg)
                        mtype = data.get("type")
                        mdata = data.get("data", {})

                        if mtype == "status":
                            q_remain = mdata.get("status", {}).get("exec_info", {}).get("queue_remaining", 0)
                            if q_remain == 0 and completed:
                                break

                        elif mtype == "executing":
                            node = mdata.get("node")
                            p_id = mdata.get("prompt_id")
                            if p_id == prompt_id:
                                if node is None:
                                    completed = True
                                    break
                                else:
                                    node_info = prompt_dict.get(str(node), {})
                                    c_name = node_info.get("class_type", f"Node {node}")
                                    title = node_info.get("_meta", {}).get("title", c_name)
                                    log_cb(f"⚙️ Menjalankan: {title} ({c_name})")

                        elif mtype == "progress":
                            val = mdata.get("value", 0)
                            total = mdata.get("max", 1)
                            frac = min(1.0, val / max(1, total))
                            if progress_cb:
                                progress_cb(frac, f"Sampling: {val}/{total}")
                            elif val % 2 == 0 or val == total:
                                log_cb(f"⏳ Sampling: {val}/{total} steps ({frac * 100:.0f}%)")

                        elif mtype == "execution_error":
                            err_node = mdata.get("node_id")
                            err_msg = mdata.get("exception_message", "Unknown error")
                            raise ComfyAPIError(f"Error pada node {err_node}: {err_msg}")

                except websocket.WebSocketTimeoutException:
                    pass

            else:
                # HTTP Polling fallback
                time.sleep(2.0)
                h_url = f"http://{host}:{port}/history/{prompt_id}"
                try:
                    with urllib.request.urlopen(h_url) as resp:
                        history = json.loads(resp.read().decode("utf-8"))
                        if prompt_id in history:
                            completed = True
                            break
                except Exception:
                    pass

    finally:
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

    log_cb("Eksekusi workflow selesai! Mengambil file video hasil...")
    # Get history to find written video files
    h_url = f"http://{host}:{port}/history/{prompt_id}"
    with urllib.request.urlopen(h_url) as resp:
        history = json.loads(resp.read().decode("utf-8"))

    prompt_res = history.get(prompt_id, {})
    outputs = prompt_res.get("outputs", {})

    c_root = Path(comfy_root).resolve()
    found_videos = []
    for node_out in outputs.values():
        for v_item in node_out.get("videos", node_out.get("gifs", [])):
            fname = v_item.get("filename")
            subf = v_item.get("subfolder", "")
            if fname:
                found_videos.append((subf, fname))

    if not found_videos:
        # Fallback search in ComfyUI/output
        comfy_out = c_root / "output"
        if comfy_out.is_dir():
            mp4s = sorted(comfy_out.glob("**/*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
            if mp4s:
                found_videos.append(("", mp4s[0].name))

    if not found_videos:
        raise ComfyAPIError("Tidak ditemukan file video hasil dari eksekusi ComfyUI.")

    # Select best video: second_pass > first_pass > comparison > first item
    selected = None
    for pref in ["second_pass", "first_pass", "head_swap", "comparison"]:
        for subf, fname in found_videos:
            if pref in fname.lower():
                selected = (subf, fname)
                break
        if selected:
            break

    if not selected:
        selected = found_videos[0]

    subfolder, filename = selected
    out_video_path = c_root / "output" / subfolder / filename

    log_cb(f"Video hasil terpilih: {out_video_path.name}")
    for subf, fname in found_videos:
        fpath = c_root / "output" / subf / fname
        if fpath.is_file():
            log_cb(f"  🎬 File tersedia: {fpath.name} ({fpath.stat().st_size / (1024*1024):.2f} MB)")

    if dest_file:
        dst = Path(dest_file).resolve()
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_video_path, dst)
        return dst

    return out_video_path
