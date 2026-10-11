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


def is_comfy_running(host: str = "127.0.0.1", port: int = 8188) -> bool:
    """Check if ComfyUI server is actively responding on port 8188."""
    url = f"http://{host}:{port}/system_stats"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MosaicDiff-Runner"})
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def start_comfy_daemon(
    comfy_root: str | Path = "/content/ComfyUI",
    python_bin: str = sys.executable,
    host: str = "127.0.0.1",
    port: int = 8188,
    log_cb=print,
    timeout_sec: int = 60,
) -> subprocess.Popen | None:
    """Ensure ComfyUI backend server is running in background."""
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


def convert_workflow_to_prompt(
    workflow_path: str | Path,
    host: str = "127.0.0.1",
    port: int = 8188,
    log_cb=None,
) -> dict:
    """Load API prompt format for execution.
    
    Prefers pre-compiled .api.json file directly from disk, eliminating network calls,
    conversion errors, and any dependency on server-side converter endpoints.
    """
    wf_path = Path(workflow_path).resolve()

    # 1. Check for pre-compiled companion .api.json file
    api_candidates = [
        wf_path.with_name(wf_path.stem + ".api.json"),
        wf_path.parent / (wf_path.name.replace(".json", ".api.json")),
        Path(__file__).resolve().parent.parent / "workflow_ltx2.5_v1.1_head_swap_drag_and_drop.api.json",
        Path("/content/mosdif/workflow_ltx2.5_v1.1_head_swap_drag_and_drop.api.json"),
    ]
    for cand in api_candidates:
        if cand.is_file():
            try:
                with open(cand, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and any("class_type" in v for v in data.values()):
                    if log_cb:
                        log_cb(f"Memuat pre-compiled API prompt: {cand.name}")
                    return data
            except Exception:
                pass

    if not wf_path.is_file():
        raise FileNotFoundError(f"Workflow file tidak ditemukan: {wf_path}")

    with open(wf_path, "r", encoding="utf-8") as f:
        wf_json = json.load(f)

    # 2. Check if the provided file is already in API format
    if isinstance(wf_json, dict) and any("class_type" in v for v in wf_json.values()):
        return wf_json

    # 3. Try server-side converter endpoint (comfyui-workflow-to-api-converter-endpoint)
    conv_url = f"http://{host}:{port}/workflow/convert"
    try:
        data = json.dumps(wf_json).encode("utf-8")
        req = urllib.request.Request(
            conv_url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            if isinstance(res, dict) and "prompt" in res:
                return res["prompt"]
            if isinstance(res, dict) and any("class_type" in v for v in res.values()):
                return res
    except urllib.error.HTTPError as http_err:
        err_msg = http_err.read().decode("utf-8", errors="replace")
        raise ComfyAPIError(f"ComfyUI /workflow/convert gagal (HTTP {http_err.code}):\n{err_msg}") from http_err
    except Exception as exc:
        raise ComfyAPIError(f"Gagal menghubungi /workflow/convert: {exc}") from exc

    raise ComfyAPIError(
        f"Gagal mengonversi workflow UI ke format API prompt.\n"
        "Pastikan node 'comfyui-workflow-to-api-converter-endpoint' sudah terpasang di ComfyUI/custom_nodes/."
    )


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
    video_input_str = str(Path(video_input_path).resolve())

    # Format BFS Head Swap prompt
    p_text = prompt_text.strip()
    if not p_text.lower().startswith("head_swap:") and not p_text.lower().startswith("head swap"):
        if p_text:
            pos_prompt = f"head_swap:\nFACE:\n{p_text}"
        else:
            pos_prompt = "head_swap:\nFACE:\nPhotorealistic human face, natural skin texture, sharp detailed eyes, realistic lighting"
    else:
        pos_prompt = p_text

    neg_prompt = negative_prompt_text.strip() or "mouth movement, cartoon, anime, illustration, 3d render, plastic, painting, blurry, deformed"

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
            # Sync lora_name to existing file
            cur_lora = inputs.get("lora_name", "")
            if cur_lora not in available_loras:
                for cand in available_loras:
                    if "head_swap" in cand.lower():
                        inputs["lora_name"] = cand
                        break

        # 4. Positive Prompt (CLIPTextEncode, Node 448)
        if ctype == "CLIPTextEncode" and ("Positive" in title or "head_swap" in str(inputs.get("text", "")).lower()):
            inputs["text"] = pos_prompt

        # 5. Negative Prompt (CLIPTextEncode, Node 444)
        if ctype == "CLIPTextEncode" and ("Negative" in title or "mouth movement" in str(inputs.get("text", "")).lower()):
            inputs["text"] = neg_prompt

        # 6. Samplers & Schedulers
        if "seed" in inputs or "noise_seed" in inputs:
            k = "noise_seed" if "noise_seed" in inputs else "seed"
            inputs[k] = int(seed)

        if ctype in ("BasicScheduler", "LTXVScheduler") and "steps" in inputs:
            inputs["steps"] = int(steps)

        # 7. Second Pass & Comparison Video Handling
        if not enable_second_pass:
            if ctype == "SaveVideo" and "Second Pass" in title:
                nodes_to_remove.append(nid)
            elif ctype == "VHS_VideoCombine" and ("comparison" in str(inputs).lower() or "comparison" in title.lower()):
                nodes_to_remove.append(nid)

    # Prune second pass save nodes if disabled so ComfyUI doesn't execute them
    for nid in nodes_to_remove:
        patched.pop(nid, None)

    # 8. Clean up Qwen auto-prompt subgraph if Qwen3-VL model is not downloaded
    qwen_model_path = Path(comfy_root).resolve() / "models" / "text_encoders" / "qwen3-vl-4b-heretic_int8_dynamic_convrot.safetensors"
    if not qwen_model_path.is_file():
        qwen_nodes = ["445", "447", "607", "608", "602", "563", "645", "567"]
        for qnid in qwen_nodes:
            patched.pop(qnid, None)
        if "600" in patched and "inputs" in patched["600"]:
            patched["600"]["inputs"]["text"] = "BFS Head Swap"

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
