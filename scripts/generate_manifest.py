import os
import json
import hashlib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
artifacts_dir = os.path.join(PROJECT_ROOT, "ml", "artifacts")

files_to_hash = {
    "model.pkl": "model.pkl",
    "pipeline.pkl": "pipeline.pkl",
    "encoders.pkl": "encoders.pkl",
    "model_metadata.pkl": "model_metadata.pkl",
    "kmeans_model.pkl": os.path.join("models", "kmeans_model.pkl"),
    "autoencoder_model.pkl": os.path.join("models", "autoencoder_model.pkl")
}

manifest = {}

for name, rel_path in files_to_hash.items():
    filepath = os.path.join(artifacts_dir, rel_path)
    if os.path.exists(filepath):
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        manifest[name] = sha256.hexdigest()
        print(f"Hashed {name}: {manifest[name]}")
    else:
        print(f"Warning: File not found at {filepath}")

manifest_path = os.path.join(artifacts_dir, "artifacts_manifest.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=4)
print(f"Manifest written to {manifest_path}")
