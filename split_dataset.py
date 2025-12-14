import os
import shutil
import random

def diviser_dataset_simple(chemin_dataset, chemin_output, ratio_test=0.2):
    """
    Division simple sans scikit-learn
    """
    
    # Créer les dossiers
    train_dir = os.path.join(chemin_output, 'train')
    test_dir = os.path.join(chemin_output, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    classes = os.listdir(chemin_dataset)
    
    for classe in classes:
        if not os.path.isdir(os.path.join(chemin_dataset, classe)):
            continue
            
        print(f"Traitement: {classe}")
        
        chemin_classe = os.path.join(chemin_dataset, classe)
        images = [f for f in os.listdir(chemin_classe) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Mélanger les images
        random.shuffle(images)
        
        # Calculer le point de coupure
        split_index = int(len(images) * (1 - ratio_test))
        
        train_images = images[:split_index]
        test_images = images[split_index:]
        
        # Créer les dossiers de classe
        os.makedirs(os.path.join(train_dir, classe), exist_ok=True)
        os.makedirs(os.path.join(test_dir, classe), exist_ok=True)
        
        # Copier les images
        for img in train_images:
            shutil.copy2(os.path.join(chemin_classe, img), 
                        os.path.join(train_dir, classe, img))
        
        for img in test_images:
            shutil.copy2(os.path.join(chemin_classe, img), 
                        os.path.join(test_dir, classe, img))
        
        print(f"  ✓ Train: {len(train_images)}, Test: {len(test_images)}")

# Utilisation rapide
if __name__ == "__main__":
    diviser_dataset_simple(
        chemin_dataset="Dataset\dataset_equilibre",
        chemin_output="Dataset/dataset_for_train",
        ratio_test=0.2  # 20% test
    )