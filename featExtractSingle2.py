import os, csv, numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import (
    VGG16, InceptionV3, ResNet50,
    vgg16, inception_v3, resnet50
)

ROOT_DATASETS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
OUTPUT_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))

MODELS_CFG = {
    "vgg16":       (lambda: VGG16(weights="imagenet", include_top=False, pooling="avg"),
                    (224, 224), vgg16.preprocess_input),
    "resnet50":    (lambda: ResNet50(weights="imagenet", include_top=False, pooling="avg"),
                    (224, 224), resnet50.preprocess_input),
    "inceptionv3": (lambda: InceptionV3(weights="imagenet", include_top=False, pooling="avg"),
                    (299, 299), inception_v3.preprocess_input),
}

def extract(img_path, model, preprocess, target):
    img = load_img(img_path, target_size=target)
    arr = img_to_array(img)[None, ...]
    arr = preprocess(arr)
    return model.predict(arr, verbose=0).flatten()

def process_dataset(ds_path):
    ds_name = os.path.basename(ds_path.rstrip("/"))
    print(f"\n Dataset : {ds_name}")

    results = {m: [] for m in MODELS_CFG}
    for label in os.listdir(ds_path):
        class_dir = os.path.join(ds_path, label)
        if not os.path.isdir(class_dir): continue
        print(f"   • Classe : {label}")
        for img_name in os.listdir(class_dir):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(class_dir, img_name)
            for m_name, (factory, size, prep) in MODELS_CFG.items():
                if m_name not in _MODELS_CACHE:
                    print(f"      → Chargement modèle {m_name.upper()}")
                    _MODELS_CACHE[m_name] = factory()
                vec = extract(img_path, _MODELS_CACHE[m_name], prep, size)
                results[m_name].append(list(vec) + [label])
    return ds_name, results

def save_outputs(dataset_name, results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for m_name, rows in results.items():
        if not rows: continue
        n_feat = len(rows[0]) - 1
        header = [f"feature_{i+1}" for i in range(n_feat)] + ["label"]

        csv_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_{m_name}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"{m_name.upper()} à {csv_path} ({len(rows)} images)")

        # Sauvegarde NumPy
        npy_path = csv_path.replace(".csv", ".npy")
        np.save(npy_path, np.array([r[:-1] for r in rows]))
        # (labels pourraient aussi être sauvegardés séparément si besoin)

def main():
    global _MODELS_CACHE
    _MODELS_CACHE = {}

    if not os.path.isdir(ROOT_DATASETS):
        raise FileNotFoundError(f"Dossier racine introuvable : {ROOT_DATASETS}")

    for ds in sorted(os.listdir(ROOT_DATASETS)):
        ds_path = os.path.join(ROOT_DATASETS, ds)
        if not os.path.isdir(ds_path): continue
        ds_name, feats = process_dataset(ds_path)
        save_outputs(ds_name, feats)

    print("\n Tous les jeux de données ont été traités.")

if __name__ == "__main__":
    main()
