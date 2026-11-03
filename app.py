"""Launch with: python -m streamlit run app.py."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError


MODEL_DIR = Path(__file__).resolve().parent / "model"
PREPROCESSING = {
    "MobileNetV2 : pixels entre -1 et 1": "mobilenet",
    "Division par 255 : pixels entre 0 et 1": "rescale",
    "Pixels bruts : entre 0 et 255": "raw",
}


@st.cache_resource(show_spinner="Chargement du modèle…")
def load_model(config_text, weights_version):
    # The supplied folder is an extracted Keras archive.
    from tensorflow import keras

    config = json.loads(config_text)
    config.pop("compile_config", None)
    model = keras.models.model_from_json(json.dumps(config))
    model.load_weights(str(MODEL_DIR / "model.weights.h5"))
    return model


def prepare_image(image, height, width, preprocessing):
    resized = image.resize((width, height), Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32)
    if preprocessing == "mobilenet":
        pixels = pixels / 127.5 - 1.0
    elif preprocessing == "rescale":
        pixels = pixels / 255.0
    return np.expand_dims(pixels, axis=0)


def main():
    st.set_page_config(page_title="Classification d’images", page_icon="📷", layout="wide")
    st.title("📷 Classification d’images")
    st.write("Importez une photo et découvrez les prédictions du modèle.")

    try:
        config_text = (MODEL_DIR / "config.json").read_text(encoding="utf-8")
        config = json.loads(config_text)
        layers = config["config"]["layers"]
        _, height, width, channels = layers[0]["config"]["batch_shape"]
        class_count = layers[-1]["config"]["units"]
        if channels != 3 or not height or not width:
            raise ValueError("Le modèle doit accepter des images RGB de taille fixe.")
        weights_version = (MODEL_DIR / "model.weights.h5").stat().st_mtime_ns
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        st.error(f"Impossible de lire le modèle dans le dossier model : {exc}")
        st.stop()

    with st.sidebar:
        st.header("Paramètres")
        st.caption(f"MobileNetV2 · {width} × {height} pixels · {class_count} classes")
        preprocessing_label = st.selectbox("Prétraitement", list(PREPROCESSING))
        st.caption(
            "MobileNetV2 est proposé par défaut. Choisissez exactement la même "
            "normalisation que pendant l’entraînement."
        )
        default_names = [f"Classe {i}" for i in range(class_count)]
        labels_path = MODEL_DIR / "class_names.json"
        if labels_path.exists():
            try:
                saved_names = json.loads(labels_path.read_text(encoding="utf-8"))
                if not isinstance(saved_names, list) or not all(
                    isinstance(name, str) for name in saved_names
                ):
                    raise ValueError("Une liste de noms est attendue.")
                default_names = saved_names
            except (OSError, ValueError) as exc:
                st.warning(f"Impossible de lire class_names.json : {exc}")
        names_text = st.text_area(
            "Noms des classes, dans l’ordre d’entraînement",
            value="\n".join(default_names),
            height=220,
            help="Un nom par ligne, à partir de l’indice 0. Utilisez train_dataset.class_names.",
        )
        names = [name.strip() for name in names_text.splitlines() if name.strip()]
        valid_names = len(names) == class_count and len(set(names)) == class_count
        if not valid_names:
            st.error(f"Saisissez exactement {class_count} noms distincts.")

    if names == [f"Classe {i}" for i in range(class_count)]:
        st.info(
            "Les noms des classes ne sont pas fournis avec le modèle. "
            "Renseignez-les dans la barre latérale pour identifier les catégories."
        )

    uploaded = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded is None:
        st.caption("Formats acceptés : JPG, PNG et WebP.")
        return
    try:
        image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        st.error("Cette image ne peut pas être lue. Essayez un autre fichier.")
        return

    preview, results = st.columns(2, gap="large")
    with preview:
        st.subheader("Image importée")
        st.image(image, caption=uploaded.name, width="stretch")

    with results:
        st.subheader("Résultats")
        if not st.button("Analyser l’image", type="primary", disabled=not valid_names):
            return
        try:
            model = load_model(config_text, weights_version)
            batch = prepare_image(image, height, width, PREPROCESSING[preprocessing_label])
            with st.spinner("Analyse en cours…"):
                probabilities = np.asarray(model.predict(batch, verbose=0))[0]
            if probabilities.shape != (class_count,) or not np.all(np.isfinite(probabilities)):
                raise ValueError("Les scores retournés par le modèle sont invalides.")
        except ModuleNotFoundError as exc:
            st.error(f"Dépendance manquante : {exc.name}.")
            st.code('python -m pip install tensorflow "keras>=3.13.2"', language="bash")
            return
        except Exception as exc:
            st.error(f"L’analyse a échoué : {exc}")
            return

        winner = int(np.argmax(probabilities))
        st.success(f"Prédiction : **{names[winner]}**")
        st.metric("Score du modèle", f"{probabilities[winner]:.1%}")
        scores = pd.DataFrame({"Classe": names, "Score (%)": probabilities * 100})
        scores = scores.sort_values("Score (%)", ascending=False)
        st.bar_chart(scores.set_index("Classe"), horizontal=True, sort=False)
        st.dataframe(scores, hide_index=True, width="stretch")


if __name__ == "__main__":
    main()
