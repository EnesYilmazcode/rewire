"""Run steer.py on Vertex AI spot A100s, one job per model, all in parallel.

The project has 8 spot A100s in each of us-central1, europe-west4 and asia-southeast1. Code goes to the bucket,
each job installs transformers, runs, and writes its JSON and log back to gs://motionguys-rl-runs/rewire/out/.
Usage: python vertex.py submit            (launch every job in JOBS)
       python vertex.py status            (state of each job)
       python vertex.py fetch             (copy finished results into results/steer/)
"""
import json
import os
import subprocess
import sys
import urllib.request

PROJECT, BUCKET = "motionguys-rl", "motionguys-rl-runs"
GCLOUD = os.path.expandvars(r"%LOCALAPPDATA%\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd")
IMAGE = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest"
CODE = f"gs://{BUCKET}/rewire/code"
JOBS = [  # (model, region, A100 count)
    ("Qwen/Qwen3.5-9B", "us-central1", 1),
    ("Qwen/Qwen3.5-27B", "europe-west4", 2),
    ("Qwen/Qwen3.5-35B-A3B", "asia-southeast1", 2),  # spot CPU quota is 42 per region: 4 GPUs needs 48
]
ARGS = "--train 150 --test 150 --mode level --scales -32 -16 16 32 64"
STATE = "results/steer/vertex_jobs.json"


def gcloud(*a):
    return subprocess.run([GCLOUD, *a], capture_output=True, text=True, check=True).stdout.strip()


def call(method, url, body=None):
    token = gcloud("auth", "print-access-token")
    req = urllib.request.Request(url, json.dumps(body).encode() if body else None, method=method, headers={
        "Authorization": f"Bearer {token}", "x-goog-user-project": PROJECT, "Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def api(region):
    return f"https://{region}-aiplatform.googleapis.com/v1/projects/{PROJECT}/locations/{region}/customJobs"


def submit(jobs_list=None, command=None, suffix=""):
    """command: what to run in place of steer.py, with {model} filled in; suffix tags the job."""
    gcloud("storage", "cp", "ladder.py", "steer.py", "confirm.py", CODE + "/")
    jobs = json.load(open(STATE)) if os.path.exists(STATE) else {}
    for model, region, gpus in (jobs_list or JOBS):
        tag = model.split("/")[-1].lower() + suffix
        if tag in jobs:
            continue
        out = f"/gcs/{BUCKET}/rewire/out"
        run = (command or "python steer.py --model {model} " + ARGS).format(model=model)
        script = (f"set -xo pipefail; mkdir -p {out} /w && cp /gcs/{BUCKET}/rewire/code/*.py /w/ && cd /w && "
                  f"pip install -q -U 'torch>=2.6' --index-url https://download.pytorch.org/whl/cu124 && pip uninstall -y -q torchvision torchaudio; "
                  f"pip install -q -U transformers accelerate flash-linear-attention 'jinja2>=3.1' && nvidia-smi && "  # image ships torch 2.4
                  f"OUT={out} {run} 2>&1 | tee {out}/{tag}.log")
        body = {
            "displayName": f"rewire-{tag}",
            "jobSpec": {
                "workerPoolSpecs": [{
                    "machineSpec": {"machineType": f"a2-highgpu-{gpus}g", "acceleratorType": "NVIDIA_TESLA_A100",
                                    "acceleratorCount": gpus},
                    "replicaCount": "1",
                    "diskSpec": {"bootDiskType": "pd-ssd", "bootDiskSizeGb": 200},
                    "containerSpec": {"imageUri": IMAGE, "command": ["bash", "-c"], "args": [script]},
                }],
                "scheduling": {"strategy": "SPOT", "timeout": "14400s"},
            },
        }
        job = call("POST", api(region), body)
        jobs[tag] = {"region": region, "id": job["name"].rsplit("/", 1)[1], "model": model}
        print(f"submitted {tag} on {gpus}x A100 in {region}: {jobs[tag]['id']}")
        os.makedirs("results/steer", exist_ok=True)
        json.dump(jobs, open(STATE, "w"), indent=1)  # saved after each job, so a failure never loses track


def cancel(tag):
    j = json.load(open(STATE))[tag]
    call("POST", f"{api(j['region'])}/{j['id']}:cancel", {})


def status():
    for tag, j in json.load(open(STATE)).items():
        s = call("GET", f"{api(j['region'])}/{j['id']}")
        print(f"{tag:22s} {s['state'].replace('JOB_STATE_', ''):10s} {s.get('error', {}).get('message', '')[:150]}")


def fetch():
    os.makedirs("results/steer", exist_ok=True)
    print(gcloud("storage", "cp", f"gs://{BUCKET}/rewire/out/*", "results/steer/"))


if __name__ == "__main__":
    {"submit": submit, "status": status, "fetch": fetch}[sys.argv[1]]()
