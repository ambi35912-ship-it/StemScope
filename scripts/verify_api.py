"""Exercise a running API with real audio; retain a small JSON verification record."""

import argparse
import io
import json
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--report", type=Path, default=Path("outputs/api-verification.json"))
    args = parser.parse_args()
    expected = sf.info(args.audio)
    results = []
    with httpx.Client(base_url=args.url, timeout=600, trust_env=False) as client:
        health = client.get("/health")
        health.raise_for_status()
        models = client.get("/models")
        models.raise_for_status()
        for model in models.json()["models"]:
            with args.audio.open("rb") as stream:
                response = client.post(
                    "/separations",
                    data={"model": model},
                    files={"audio": (args.audio.name, stream, "audio/wav")},
                )
            response.raise_for_status()
            result = response.json()
            assert set(result["stems"]) == {"vocals", "drums", "bass", "other"}
            for url in result["stems"].values():
                download = client.get(url)
                download.raise_for_status()
                samples, rate = sf.read(io.BytesIO(download.content), always_2d=True)
                assert len(samples) > 0 and np.isfinite(samples).all()
                assert abs(len(samples) / rate - expected.duration) <= 0.1
            results.append(result)
            print(f"Passed: {model}, four valid WAV downloads", flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps({"health": health.json(), "results": results}, indent=2) + "\n"
    )
    print(f"Saved {args.report}")


if __name__ == "__main__":
    main()
