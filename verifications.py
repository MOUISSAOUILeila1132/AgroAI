import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def analyser_equilibre_classes(chemin_dataset):
    """
    Analyse l'équilibre des classes dans un dataset d'images
    
    Args:
        chemin_dataset (str): Chemin vers le dossier principal du dataset
    """
    
    # Vérifier si le dossier existe
    if not os.path.exists(chemin_dataset):
        print(f"Erreur: Le dossier {chemin_dataset} n'existe pas.")
        return
    
    # Lister les classes (sous-dossiers)
    classes = [d for d in os.listdir(chemin_dataset) 
               if os.path.isdir(os.path.join(chemin_dataset, d))]
    
    if not classes:
        print("Aucune classe trouvée. Vérifiez la structure du dataset.")
        return
    
    print(f"Nombre de classes trouvées: {len(classes)}")
    print(f"Classes: {classes}")
    print("\n" + "="*50)
    
    # Compter les images par classe
    compteur_images = {}
    total_images = 0
    
    for classe in classes:
        chemin_classe = os.path.join(chemin_dataset, classe)
        
        # Lister les fichiers image (extensions courantes)
        extensions_images = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        images = [f for f in os.listdir(chemin_classe) 
                 if os.path.isfile(os.path.join(chemin_classe, f)) and 
                 any(f.lower().endswith(ext) for ext in extensions_images)]
        
        compteur_images[classe] = len(images)
        total_images += len(images)
    
    # Afficher les résultats
    print("\nRÉSULTATS DE L'ANALYSE:")
    print("="*50)
    
    for classe, count in compteur_images.items():
        pourcentage = (count / total_images) * 100
        print(f"{classe}: {count} images ({pourcentage:.2f}%)")
    
    print(f"\nTotal d'images: {total_images}")
    
    # Calculer les métriques d'équilibre
    counts = list(compteur_images.values())
    min_count = min(counts)
    max_count = max(counts)
    ratio_equilibre = min_count / max_count if max_count > 0 else 0
    
    print(f"\nMÉTRIQUES D'ÉQUILIBRE:")
    print(f"Classe la plus petite: {min_count} images")
    print(f"Classe la plus grande: {max_count} images")
    print(f"Ratio d'équilibre: {ratio_equilibre:.3f}")
    
    # Déterminer si le dataset est équilibré
    if ratio_equilibre > 0.8:
        equilibre = "TRÈS ÉQUILIBRÉ"
    elif ratio_equilibre > 0.6:
        equilibre = "MODÉRÉMENT ÉQUILIBRÉ"
    elif ratio_equilibre > 0.4:
        equilibre = "PEU ÉQUILIBRÉ"
    else:
        equilibre = "DÉSÉQUILIBRÉ"
    
    print(f"Évaluation: {equilibre}")
    
    # Visualisation
    visualiser_equilibre(compteur_images)

def visualiser_equilibre(compteur_images):
    """
    Crée des visualisations pour l'équilibre des classes
    """
    classes = list(compteur_images.keys())
    counts = list(compteur_images.values())
    
    # Créer une figure avec deux sous-graphiques
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Graphique à barres
    bars = ax1.bar(classes, counts, color='skyblue', edgecolor='black')
    ax1.set_title('Distribution des images par classe', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Classes')
    ax1.set_ylabel('Nombre d\'images')
    ax1.tick_params(axis='x', rotation=45)
    
    # Ajouter les valeurs sur les barres
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(count), ha='center', va='bottom')
    
    # Camembert
    colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
    wedges, texts, autotexts = ax2.pie(counts, labels=classes, autopct='%1.1f%%',
                                      colors=colors, startangle=90)
    ax2.set_title('Répartition en pourcentage', fontsize=14, fontweight='bold')
    
    # Améliorer l'apparence du camembert
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    plt.show()
    
    # Graphique de ratio d'équilibre
    fig, ax = plt.subplots(figsize=(10, 6))
    ratio_ideal = [max(counts)] * len(classes)  # Ligne de référence pour l'équilibre parfait
    ax.plot(classes, counts, 'o-', label='Distribution actuelle', linewidth=2, markersize=8)
    ax.plot(classes, ratio_ideal, 'r--', label='Équilibre parfait', alpha=0.7)
    ax.set_title('Comparaison avec l\'équilibre parfait', fontsize=14, fontweight='bold')
    ax.set_xlabel('Classes')
    ax.set_ylabel('Nombre d\'images')
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def analyser_structure_dataset(chemin_dataset):
    """
    Analyse la structure du dataset et vérifie les formats de fichiers
    """
    print("ANALYSE DE LA STRUCTURE DU DATASET:")
    print("="*50)
    
    classes = [d for d in os.listdir(chemin_dataset) 
               if os.path.isdir(os.path.join(chemin_dataset, d))]
    
    for classe in classes:
        chemin_classe = os.path.join(chemin_dataset, classe)
        fichiers = os.listdir(chemin_classe)
        
        if not fichiers:
            print(f"⚠️  Attention: La classe '{classe}' est vide!")
            continue
        
        # Analyser les extensions de fichiers
        extensions = {}
        for fichier in fichiers:
            _, ext = os.path.splitext(fichier)
            ext = ext.lower()
            extensions[ext] = extensions.get(ext, 0) + 1
        
        print(f"\nClasse: {classe}")
        for ext, count in extensions.items():
            print(f"  {ext}: {count} fichiers")

# Utilisation du script
if __name__ == "__main__":
    # Remplacez par le chemin de votre dataset
    chemin_votre_dataset = "Dataset\combined_dataset"
    
    # Analyser la structure
    analyser_structure_dataset(chemin_votre_dataset)
    
    # Analyser l'équilibre
    analyser_equilibre_classes(chemin_votre_dataset)