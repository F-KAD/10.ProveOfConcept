import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.preprocessing.image import load_img

# Chemin vers les dossiers d'images
base_image_path = "images"
train_path = os.path.join(base_image_path, "train")
test_path = os.path.join(base_image_path, "test")
val_path = os.path.join(base_image_path, "val")

# Labels et mappage
label2id = {"interior": 0, "outside": 1, "food": 2, "drink": 3, "menu": 4}
id2label = {value: key for key, value in label2id.items()}

# Génération des DataFrames
def generate_dataframe(base_path, label2id):
    data = []
    for label, label_id in label2id.items():
        folder_path = os.path.join(base_path, label)
        for image_name in os.listdir(folder_path):
            if image_name.endswith(".jpg") or image_name.endswith(".png"):
                img_path = os.path.join(folder_path, image_name)
                data.append({
                    "photo_id": image_name,
                    "label": label,
                    "label_num": label_id,
                    "img_path": img_path,
                })
    df = pd.DataFrame(data)
    return df

df_train = generate_dataframe(train_path, label2id)
df_test = generate_dataframe(test_path, label2id)
df_val = generate_dataframe(val_path, label2id)

# Fusionner les DataFrames pour faciliter l'accès
df_all = pd.concat([df_train, df_test, df_val], ignore_index=True)

# Fonction pour prédire les labels (à remplacer par votre modèle de prédiction)
def predict_label(img_path):
    # Implémentez ici la logique de prédiction avec votre modèle
    # Exemple : model.predict(...)
    return random.choice(list(label2id.values()))

# Ajouter les prédictions aux DataFrames
df_all["label_num_classique"] = df_all["img_path"].apply(predict_label)
df_all["label_num_adverse"] = df_all["img_path"].apply(predict_label)

# Fonction pour ajuster la taille des textes
def custom_text(text, text_size, Type):
    if Type == "titre":
        st.markdown(f"<h1 style='text-align: center; color: gray; font-size:{text_size + 20}px;'>{text}</h1>", unsafe_allow_html=True)
    elif Type == "sous-titre":
        st.markdown(f"<h2 style='font-size:{text_size + 10}px;'>{text}</h1>", unsafe_allow_html=True)
    elif Type == "texte":
        st.markdown(f"<div class='custom-text-size' style='font-size:{text_size}px;'>{text}</div>", unsafe_allow_html=True)

def Render():
    st.sidebar.subheader("Navigation")
    page = st.sidebar.radio("Aller à", ["I - Présentation des données",
                                        "II - Attaques Adverses",
                                        "III - Modèle VGG16 face aux attaques adverses",
                                        "IV - Exemples d'attaques adverses sur les modèles"])
    text_size = st.sidebar.slider("Taille du texte", 10, 36, 16)
    custom_text("VGG16 et Attaques Adverses", text_size, "titre")
    st.image(image="img_notebook/dataspace.png", caption="Logo de DataSpace")

    if page == "I - Présentation des données":
        custom_text("Présentation des données", text_size, "sous-titre")
        custom_text("""Bienvenue sur la page de présentation de ce Dashboard ! Le but est d'étudier comment des attaques adverses
                       peuvent influencer un modèle de classification. Tout d'abord, vous retrouverez ci-dessous un affichage
                       d'un sample d'images en fonction de leurs labels pour avoir un premier visuel sur nos données.""",
                    text_size, "texte")
        st.write("")
        labels = st.multiselect("Sélectionnez une catégorie d'images pour avoir un aperçu",
                                ["interior", "outside", "food", "drink", "menu"])
        if labels:
            if st.button("Appuyez pour afficher"):
                for label in labels:
                    names = np.array(df_all["photo_id"][df_all["label"] == label].sample(3))
                    images = [load_img(df_all[df_all["photo_id"] == name]["img_path"].values[0], target_size=(224, 224)) for name in names]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.image(images[0], caption=f"Image {label} 1")
                    with col2:
                        st.image(images[1], caption=f"Image {label} 2")
                    with col3:
                        st.image(images[2], caption=f"Image {label} 3")

    elif page == "II - Attaques Adverses":
        custom_text("Attaques Adverses", text_size, "sous-titre")
        custom_text("""Bienvenue sur la page des Attaques Adverses de ce Dashboard ! Le but est d'étudier comment nous pouvons
                       créer des attaques adverses. Voici un exemple :""", text_size, "texte")
        st.write("")
        st.image(image="img_notebook/attaque_adverse.jpeg", caption="Exemple d'attaque adverse")
        custom_text("""Comme nous pouvons le voir ci-dessus, nous avons une image de chat (image de gauche), le modèle nous dit que
                       c'est un chat égyptien, ce qui est correct. En ajoutant une perturbation (image du milieu) multipliée par
                       un coefficient epsilon, nous avons une image qui semble identique à l'œil humain ! Or, cette fois elle a
                       été prédite comme voiture de sport... C'est ainsi que nous créons une attaque adverse.""",
                    text_size, "texte")

    elif page == "III - Modèle VGG16 face aux attaques adverses":
        custom_text("Modèle VGG16 face aux attaques adverses", text_size, "sous-titre")
        custom_text("""Bienvenue sur la page du Modèle VGG16 face aux attaques adverses de ce Dashboard ! Le but est d'étudier le
                       comportement du modèle VGG16 face à des attaques adverses. Nous avons deux modèles :""",
                    text_size, "texte")
        st.write("")
        custom_text("- Un modèle VGG16 classique : non entraîné avec des attaques adverses", text_size, "texte")
        custom_text("- Un modèle VGG16 adversarial : entraîné avec des attaques adverses", text_size, "texte")
        st.write("")
        affiche = st.selectbox("Afficher les résultats pour le modèle VGG16 :", ["Classique", "Adverse"])

        if affiche == "Classique":
            y_true = df_all["label_num"]
            y_pred = df_all["label_num_classique"]
            y_pred_adv = df_all["label_num_adverse"]
            conf_mat = confusion_matrix(y_true, y_pred)
            conf_mat_heatmap = pd.DataFrame(conf_mat, index=label2id)
            conf_mat_adv = confusion_matrix(y_true, y_pred_adv)
            conf_mat_heatmap_adv = pd.DataFrame(conf_mat_adv, index=label2id)
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            sns.heatmap(conf_mat_heatmap, annot=True, cmap="Blues", fmt='g', ax=axes[0])
            axes[0].set_title('Heatmap valeurs / prédictions classiques')
            sns.heatmap(conf_mat_heatmap_adv, annot=True, cmap="Blues", fmt='g', ax=axes[1])
            axes[1].set_title('Heatmap valeurs / prédictions adverses')
            plt.tight_layout()
            st.pyplot(fig)
            custom_text("Nous pouvons voir que le modèle n'est pas robuste aux attaques adverses", text_size, "texte")
            st.write("")
            custom_text(f"Accuracy_classique : {accuracy_score(y_true, y_pred)} / Accuracy_adverse : {accuracy_score(y_true, y_pred_adv)}", text_size, "texte")

        elif affiche == "Adverse":
            y_true = df_all["label_num"]
            y_pred = df_all["label_num_classique"]
            y_pred_adv = df_all["label_num_adverse"]
            conf_mat = confusion_matrix(y_true, y_pred)
            conf_mat_heatmap = pd.DataFrame(conf_mat, index=label2id)
            conf_mat_adv = confusion_matrix(y_true, y_pred_adv)
            conf_mat_heatmap_adv = pd.DataFrame(conf_mat_adv, index=label2id)
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            sns.heatmap(conf_mat_heatmap, annot=True, cmap="Blues", fmt='g', ax=axes[0])
            axes[0].set_title('Heatmap valeurs / prédictions classiques')
            sns.heatmap(conf_mat_heatmap_adv, annot=True, cmap="Blues", fmt='g', ax=axes[1])
            axes[1].set_title('Heatmap valeurs / prédictions adverses')
            plt.tight_layout()
            st.pyplot(fig)
            custom_text("Nous pouvons voir que le modèle n'est pas robuste aux attaques adverses", text_size, "texte")
            st.write("")
            custom_text(f"Accuracy_classique : {accuracy_score(y_true, y_pred)} / Accuracy_adverse : {accuracy_score(y_true, y_pred_adv)}", text_size, "texte")

    elif page == "IV - Exemples d'attaques adverses sur les modèles":
        custom_text("Exemples d'attaques adverses sur les modèles", text_size, "sous-titre")
        custom_text("""Sur cette page, nous allons examiner des exemples spécifiques d'images avant et après avoir subi des attaques adverses.
                       Vous pouvez sélectionner une image ci-dessous pour voir la comparaison.""", text_size, "texte")
        labels = st.multiselect("Sélectionnez une catégorie d'images pour avoir un aperçu", ["interior", "outside", "food", "drink", "menu"])
        if labels:
            if st.button("Appuyez pour afficher"):
                for label in labels:
                    names = np.array(df_all["photo_id"][df_all["label"] == label].sample(3))
                    for name in names:
                        image_classique = load_img(df_all[df_all["photo_id"] == name]["img_path"].values[0], target_size=(224, 224))
                        image_adverse = load_img(df_all[df_all["photo_id"] == name]["img_path"].values[0], target_size=(224, 224)) # Remplacez par le vrai chemin d'image adverse
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(image_classique, caption=f"Image classique - {label}")
                        with col2:
                            st.image(image_adverse, caption=f"Image adverse - {label}")

if __name__ == "__main__":
    Render()
