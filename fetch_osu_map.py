import os
import json
import zipfile
from pathlib import Path

def load_info_osz(osz_data, output_dir, prefix):
    extracted = {"audio": None, "osu_files": []}
    with zipfile.ZipFile(osz_data) as z:
        for name in z.namelist():
            lower = name.lower()
            if lower.endswith((".mp3", ".ogg")) and extracted["audio"] is None:
                audio_path = output_dir / f"audio-{prefix}{os.path.splitext(name)[1]}"
                with z.open(name) as src, open(audio_path, "wb") as dst:
                    dst.write(src.read())
                extracted["audio"] = str(audio_path)
            elif lower.endswith(".osu"):
                osu_path = output_dir / f"{prefix}-{len(extracted['osu_files'])+1}.osu"
                with z.open(name) as src, open(osu_path, "wb") as dst:
                    dst.write(src.read())
                extracted["osu_files"].append(str(osu_path))
    return extracted

def load_maps(raw_dir="raw", output_dir="maps", limit=100):
    beatmapsets = os.listdir(raw_dir)

    os.makedirs(output_dir, exist_ok=True)
    results = []

    for i, map in enumerate(beatmapsets, 1):
        set_dir = Path(output_dir) / f"set_{i}"
        set_dir.mkdir(parents=True, exist_ok=True)

        try:
            extracted = load_info_osz(Path(raw_dir).joinpath(map), set_dir, i)

            set_info = {
                "audio": extracted["audio"],
                "beatmaps": extracted["osu_files"],
            }
            results.append(set_info)
            print(f"[OK] load set {map} -> {set_dir}")
        except Exception as e:
            print(f"[FAIL] load Set {map} error: {e}")

    metadata_path = Path(output_dir) / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results

if __name__ == "__main__":
    metadata = load_maps(limit=10)
    print("Downloaded maps metadata saved to maps/metadata.json")
