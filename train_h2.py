"""
train_h2.py
Wavelet Patchify only (no SCA) — H2 baseline, and H3's control condition.
"""

from train_common import run_experiment

if __name__ == "__main__":
    run_experiment(
        use_sca=False,
        model_name="WM-ConvMixer",
        output_path="h2_results.json",
        result_key="WM-ConvMixer",
    )