"""Minimal Ollama client (standard library only)."""
import json, os, shutil, subprocess, time, urllib.error, urllib.request

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
if not HOST.startswith("http"):
    HOST = "http://" + HOST


def _req(path, payload=None, timeout=600):
    data = json.dumps(payload).encode() if payload is not None else None
    r = urllib.request.Request(HOST + path, data=data,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return resp.read()


def running():
    try:
        _req("/api/tags", timeout=3)
        return True
    except Exception:
        return False


def ensure_running(log=print):
    """Start the Ollama service if it is installed but not running."""
    if running():
        return True
    exe = shutil.which("ollama")
    if not exe:
        return False
    log("  starting Ollama...")
    flags = {"creationflags": 0x08000000} if os.name == "nt" else {}  # no console window
    subprocess.Popen([exe, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **flags)
    for _ in range(30):
        time.sleep(1)
        if running():
            return True
    return False


def installed_models():
    return [m["name"] for m in json.loads(_req("/api/tags")).get("models", [])]


def ensure_model(model, log=print):
    have = installed_models()
    if model in have or f"{model}:latest" in have:
        return
    log(f"  downloading model {model} (one time, several GB)...")
    r = urllib.request.Request(HOST + "/api/pull", data=json.dumps({"model": model}).encode(),
                               headers={"Content-Type": "application/json"})
    last = ""
    with urllib.request.urlopen(r, timeout=None) as resp:
        for line in resp:
            j = json.loads(line)
            if "error" in j:
                raise RuntimeError(f"Ollama could not pull {model}: {j['error']}")
            if j.get("total") and j.get("completed"):
                msg = f"  {j['status']}: {100 * j['completed'] // j['total']}%"
                if msg != last:
                    log(msg, end="\r")
                    last = msg
    log("")


def chat_json(model, system, prompt, schema, num_ctx=8192):
    """One request, reply forced into `schema`. Returns the parsed object."""
    payload = {
        "model": model, "stream": False, "format": schema, "think": False,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": prompt}],
        "options": {"temperature": 0, "num_ctx": num_ctx},
    }
    try:
        raw = _req("/api/chat", payload)
    except urllib.error.HTTPError as e:
        if e.code != 400:
            raise
        payload.pop("think")  # model or Ollama version without a thinking switch
        raw = _req("/api/chat", payload)
    return json.loads(json.loads(raw)["message"]["content"])
