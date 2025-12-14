import os
import shutil
import random
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import resample
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

def analyser_distribution(chemin_dataset):
    classes = [d for d in os.listdir(chemin_dataset) 
               if os.path.isdir(os.path.join(chemin_dataset, d))]
    distribution = {}
    for classe in classes:
        chemin_classe = os.path.join(chemin_dataset, classe)
        images = [f for f in os.listdir(chemin_classe) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        distribution[classe] = len(images)
    return distribution

def creer_augmentations():
    """Création des augmentations compatibles Albumentations 2.0.8"""
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.GaussianBlur(blur_limit=3, p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.CLAHE(clip_limit=4.0, p=0.3),
    ])

def augmenter_images(chemin_source, chemin_destination, nombre_cible):
    if not os.path.exists(chemin_destination):
        os.makedirs(chemin_destination)
    
    images_existantes = [f for f in os.listdir(chemin_source) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for img in images_existantes:
        shutil.copy2(os.path.join(chemin_source, img), 
                     os.path.join(chemin_destination, img))
    
    images_manquantes = nombre_cible - len(images_existantes)
    
    if images_manquantes > 0:
        print(f"Augmentation de {len(images_existantes)} à {nombre_cible} images pour {os.path.basename(chemin_source)}")
        augmenter = creer_augmentations()
        images_augmentees = 0
        compteur = 0
        
        while images_augmentees < images_manquantes:
            img_aleatoire = random.choice(images_existantes)
            chemin_img = os.path.join(chemin_source, img_aleatoire)
            
            try:
                image = Image.open(chemin_img).convert("RGB")
                image_np = np.array(image)
                augmented = augmenter(image=image_np)
                image_augmentee = augmented['image']
                
                nom_base = os.path.splitext(img_aleatoire)[0]
                extension = os.path.splitext(img_aleatoire)[1]
                nouveau_nom = f"{nom_base}_aug_{compteur:04d}{extension}"
                chemin_nouveau = os.path.join(chemin_destination, nouveau_nom)
                
                Image.fromarray(image_augmentee).save(chemin_nouveau)
                images_augmentees += 1
                compteur += 1
                
            except Exception as e:
                print(f"Erreur lors de l'augmentation de {chemin_img}: {e}")
                continue

def sous_echantillonner_classe(chemin_source, chemin_destination, nombre_cible):
    if not os.path.exists(chemin_destination):
        os.makedirs(chemin_destination)
    images = [f for f in os.listdir(chemin_source) 
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images_selectionnees = random.sample(images, nombre_cible)
    for img in images_selectionnees:
        shutil.copy2(os.path.join(chemin_source, img), 
                     os.path.join(chemin_destination, img))

def equilibrer_dataset(chemin_source, chemin_destination, strategie='specifique'):
    if not os.path.exists(chemin_destination):
        os.makedirs(chemin_destination)
    
    distribution = analyser_distribution(chemin_source)
    classes = list(distribution.keys())
    counts = list(distribution.values())
    
    print("DISTRIBUTION ORIGINALE:")
    for classe, count in distribution.items():
        print(f"  {classe}: {count} images")
    
    if strategie == 'moyenne':
        nombre_cible = int(np.mean(counts))
    elif strategie == 'mediane':
        nombre_cible = int(np.median(counts))
    elif strategie == 'sur_echantillonnage':
        nombre_cible = max(counts)
    elif strategie == 'specifique':
        nombre_cible = 1500
    else:
        nombre_cible = int(np.mean(counts))
    
    print(f"\nStratégie: {strategie}")
    print(f"Nombre cible par classe: {nombre_cible}")
    print(f"Nombre total d'images cible: {nombre_cible * len(classes)}")
    
    for classe in classes:
        print(f"\nTraitement de la classe: {classe}")
        chemin_source_classe = os.path.join(chemin_source, classe)
        chemin_dest_classe = os.path.join(chemin_destination, classe)
        
        nb_actuel = distribution[classe]
        
        if nb_actuel < nombre_cible:
            augmenter_images(chemin_source_classe, chemin_dest_classe, nombre_cible)
        elif nb_actuel > nombre_cible:
            sous_echantillonner_classe(chemin_source_classe, chemin_dest_classe, nombre_cible)
        else:
            shutil.copytree(chemin_source_classe, chemin_dest_classe)
    
    nouvelle_distribution = analyser_distribution(chemin_destination)
    print(f"\nDISTRIBUTION ÉQUILIBRÉE:")
    for classe, count in nouvelle_distribution.items():
        print(f"  {classe}: {count} images")
    
    return nouvelle_distribution

def visualiser_comparaison(distrib_avant, distrib_apres):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    classes = list(distrib_avant.keys())
    counts_avant = list(distrib_avant.values())
    
    ax1.bar(range(len(classes)), counts_avant, color='red', alpha=0.7)
    ax1.set_title('Distribution AVANT équilibrage')
    ax1.set_ylabel('Nombre d\'images')
    ax1.tick_params(axis='x', rotation=90)
    
    counts_apres = [distrib_apres[classe] for classe in classes]
    ax2.bar(range(len(classes)), counts_apres, color='green', alpha=0.7)
    ax2.set_title('Distribution APRÈS équilibrage')
    ax2.set_ylabel('Nombre d\'images')
    ax2.tick_params(axis='x', rotation=90)
    
    plt.tight_layout()
    plt.savefig('comparaison_equilibrage.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    chemin_dataset_original = "Dataset\combined_dataset" 
    chemin_dataset_equilibre = "Dataset\dataset_equilibre" 
    
    print("=== ANALYSE DU DATASET ORIGINAL ===")
    distrib_avant = analyser_distribution(chemin_dataset_original)
    
    print("\n=== ÉQUILIBRAGE DU DATASET ===")
    distrib_apres = equilibrer_dataset(
        chemin_dataset_original, 
        chemin_dataset_equilibre,
        strategie='specifique'
    )
    
    print("\n=== CRÉATION DU GRAPHIQUE DE COMPARAISON ===")
    visualiser_comparaison(distrib_avant, distrib_apres)
    
    print("\n=== RÉSUMÉ ===")
    print(f"Nombre total de classes: {len(distrib_apres)}")
    print(f"Nombre total d'images avant: {sum(distrib_avant.values())}")
    print(f"Nombre total d'images après: {sum(distrib_apres.values())}")
    print(f"Dataset équilibré sauvegardé dans: {chemin_dataset_equilibre}")
