# Classification d’images avec Streamlit

Deux applications de classification d’images avec TensorFlow/Keras :

- `app.py` : MobileNetV2, images RGB et huit classes (`airplane`, `car`, `cat`, `dog`, `flower`, `fruit`, `motorbike`, `person`). Configuration, poids et noms des classes dans `model/`.
- `main.py` : CNN pour les fruits (`Apple`, `Banana`, `Orange`), modèle `CNN_fruits.h5`. Le notebook `CNN_classification_fruits_Lab.ipynb` contient l’entraînement ; le jeu de données `fruits/` est à fournir séparément.

## Installation et lancement

Python 3.12 a été utilisé pour les essais.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Pour l’application dédiée aux fruits :

```bash
python -m streamlit run main.py
```

Dans l’application principale, importer une image puis cliquer sur **Analyser l’image**. Le prétraitement MobileNetV2 (pixels entre -1 et 1) est sélectionné par défaut. Les noms des classes et le prétraitement doivent correspondre à l’entraînement.

Le fichier `requirements.txt` contient les dépendances des interfaces. Pour exécuter le notebook d’entraînement, installer également `matplotlib` et Jupyter.

## Déploiement sur Streamlit Community Cloud

1. Se connecter à [Streamlit Community Cloud](https://share.streamlit.io/) avec le compte GitHub `ahmed-douyry`.
2. Autoriser Streamlit à accéder aux dépôts privés si le dépôt n’apparaît pas.
3. Cliquer sur **Create app**, puis **Yup, I have an app** et renseigner :

   | Paramètre | Valeur |
   | --- | --- |
   | Repository | `ahmed-douyry/streamlit-image-classification` |
   | Branch | `main` |
   | Main file path | `app.py` |
   | Python version (Advanced settings) | `3.12` |

4. Cliquer sur **Deploy** et attendre la fin de l’installation.

Les poids et les noms des classes sont déjà dans `model/`. Aucun secret ni téléchargement externe de modèle n’est nécessaire. TensorFlow est installé dans sa version CPU pour l’hébergement.

Le déploiement doit être lancé depuis le compte Streamlit du propriétaire ; le dépôt préparé ne signifie pas que l’application est déjà déployée. Voir la [documentation officielle](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

## Captures des essais

Les captures dans [`docs/screenshots/`](docs/screenshots/) proviennent d’essais réels dans Chrome, le 3 novembre 2026. Les images de chat, chien, voiture et pommes viennent du dossier Téléchargements ; la photo de moto vient de Wikimedia Commons. Les scores affichés sont ceux du modèle et ne constituent pas une mesure de précision sur un jeu de test.

![Accueil](docs/screenshots/00-accueil.png)

| Image | Prédiction | Score du modèle |
| --- | --- | --- |
| Chat | cat | 98,7 % |
| Chien | dog | 100,0 % |
| Voiture | car | 100,0 % |
| Pommes | fruit | 61,8 % |
| Moto | motorbike | 100,0 % |

### Chat

![Classification du chat : cat, score de 98,7 %](docs/screenshots/chat.png)

### Chien

![Classification du chien : dog, score de 100,0 %](docs/screenshots/chien.png)

### Voiture

![Classification de la voiture : car, score de 100,0 %](docs/screenshots/voiture.png)

### Pommes

![Classification des pommes : fruit, score de 61,8 %](docs/screenshots/pommes.png)

### Moto (motorbike)

![Classification de la moto : motorbike, score de 100,0 %](docs/screenshots/motorbike.png)

Photo téléchargée : [Honda XL1000V Varadero side view.jpg](https://commons.wikimedia.org/wiki/File:Honda_XL1000V_Varadero_side_view.jpg), par S de Santi, sous licence [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/). Le fichier téléchargé est conservé sans modification dans [`docs/test-images/motorbike.jpg`](docs/test-images/motorbike.jpg).

Ces cinq essais concernent `app.py`, avec le prétraitement MobileNetV2 par défaut.

Les prédictions et scores sont enregistrés dans [`results.json`](docs/screenshots/results.json).
