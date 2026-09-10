import json
import numpy as np
from sklearn.model_selection import train_test_split
from elpv_dataset.utils import load_dataset


# ============================================================
# Reproducibility and split proportions
# ============================================================

SEED = 42

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15


# ============================================================
# ELPV ordinal severity definition
# ============================================================

CLASS_NAMES = [
    "Healthy",
    "Mild",
    "Moderate",
    "Severe"
]

PROBA_TO_CLASS = {
    0.0: 0,
    round(1 / 3, 6): 1,
    round(2 / 3, 6): 2,
    1.0: 3
}


def build_locked_split():
    """
    Create one fixed train/validation/test split for ELPV.

    The split is:
        70% train
        15% validation
        15% test

    Stratification is performed using the four ordinal severity
    classes so that class proportions are approximately preserved.

    IMPORTANT:
    The public ELPV package does not provide trustworthy physical
    module identifiers. Therefore, this split is performed at the
    cell level and this limitation is explicitly recorded in the
    manifest.
    """

    # --------------------------------------------------------
    # Load the official ELPV dataset
    # --------------------------------------------------------

    images, proba, types = load_dataset()

    # Convert ELPV defect probabilities into four ordinal classes
    labels = np.array(
        [
            PROBA_TO_CLASS[round(float(p), 6)]
            for p in proba
        ],
        dtype=np.int64
    )

    indices = np.arange(len(images))

    # --------------------------------------------------------
    # First split:
    # 70% train
    # 30% temporary remainder
    # --------------------------------------------------------

    train_idx, rest_idx, train_y, rest_y = train_test_split(
        indices,
        labels,
        test_size=(VAL_FRAC + TEST_FRAC),
        stratify=labels,
        random_state=SEED
    )

    # --------------------------------------------------------
    # Second split:
    # 15% validation
    # 15% test
    #
    # The temporary remainder is split in the correct ratio:
    #
    # validation : test = 0.15 : 0.15 = 1 : 1
    # --------------------------------------------------------

    val_idx, test_idx = train_test_split(
        rest_idx,
        test_size=TEST_FRAC / (VAL_FRAC + TEST_FRAC),
        stratify=rest_y,
        random_state=SEED
    )

    # --------------------------------------------------------
    # Create locked manifest
    # --------------------------------------------------------

    manifest = {
        "seed": SEED,

        "train_fraction": TRAIN_FRAC,
        "validation_fraction": VAL_FRAC,
        "test_fraction": TEST_FRAC,

        "split_level": "cell",

        "limitation": (
            "The public ELPV release accessed through "
            "elpv_dataset.utils.load_dataset exposes image data, "
            "defect probability, and mono/poly type, but does not "
            "provide a trustworthy physical module identifier. "
            "Therefore, module-level grouping cannot be implemented "
            "from this public release. The split is consequently "
            "stratified at the cell level, and this limitation is "
            "reported explicitly."
        ),

        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist()
    }

    # --------------------------------------------------------
    # Save the locked split
    # --------------------------------------------------------

    with open("split_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # --------------------------------------------------------
    # Sanity check: class balance
    # --------------------------------------------------------

    print("===== LOCKED SPLIT CREATED =====")

    for name, idx in [
        ("train", train_idx),
        ("val", val_idx),
        ("test", test_idx)
    ]:
        counts = np.bincount(
            labels[idx],
            minlength=len(CLASS_NAMES)
        )

        print(
            f"{name}: "
            + ", ".join(
                f"{CLASS_NAMES[i]}={counts[i]}"
                for i in range(len(CLASS_NAMES))
            )
        )

    print("\nTotal samples:", len(images))
    print("Manifest saved as: split_manifest.json")

    return manifest


if __name__ == "__main__":
    build_locked_split()