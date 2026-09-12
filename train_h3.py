"""
train_h3.py
Wavelet Patchify + SCA — the H3 experiment.
"""

from train_common import run_experiment

if __name__ == "__main__":
    run_experiment(
        use_sca=True,
        model_name="WM-ConvMixer+SCA",
        output_path="h3_results.json",
        result_key="WM-ConvMixer+SCA",
    )