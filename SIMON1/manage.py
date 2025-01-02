# Dont runn!!!

# Englisch | Peharge: This source code is released under the MIT License.
#
# Usage Rights:
# The source code may be copied, modified, and adapted to individual requirements.
# Users are permitted to use this code in their own projects, both for private and commercial purposes.
# However, it is recommended to modify the code only if you have sufficient programming knowledge,
# as changes could cause unintended errors or security risks.
#
# Dependencies and Additional Frameworks:
# The code relies on the use of various frameworks and executes additional files.
# Some of these files may automatically install further dependencies required for functionality.
# It is strongly recommended to perform installation and configuration in an isolated environment
# (e.g., a virtual environment) to avoid potential conflicts with existing software installations.
#
# Disclaimer:
# Use of the code is entirely at your own risk.
# Peharge assumes no liability for damages, data loss, system errors, or other issues
# that may arise directly or indirectly from the use, modification, or redistribution of the code.
#
# Please read the full terms of the MIT License to familiarize yourself with your rights and obligations.

# Deutsch | Peharge: Dieser Quellcode wird unter der MIT-Lizenz veröffentlicht.
#
# Nutzungsrechte:
# Der Quellcode darf kopiert, bearbeitet und an individuelle Anforderungen angepasst werden.
# Nutzer sind berechtigt, diesen Code in eigenen Projekten zu verwenden, sowohl für private als auch kommerzielle Zwecke.
# Es wird jedoch empfohlen, den Code nur dann anzupassen, wenn Sie über ausreichende Programmierkenntnisse verfügen,
# da Änderungen unbeabsichtigte Fehler oder Sicherheitsrisiken verursachen könnten.
#
# Abhängigkeiten und zusätzliche Frameworks:
# Der Code basiert auf der Nutzung verschiedener Frameworks und führt zusätzliche Dateien aus.
# Einige dieser Dateien könnten automatisch weitere Abhängigkeiten installieren, die für die Funktionalität erforderlich sind.
# Es wird dringend empfohlen, die Installation und Konfiguration in einer isolierten Umgebung (z. B. einer virtuellen Umgebung) durchzuführen,
# um mögliche Konflikte mit bestehenden Softwareinstallationen zu vermeiden.
#
# Haftungsausschluss:
# Die Nutzung des Codes erfolgt vollständig auf eigene Verantwortung.
# Peharge übernimmt keinerlei Haftung für Schäden, Datenverluste, Systemfehler oder andere Probleme,
# die direkt oder indirekt durch die Nutzung, Modifikation oder Weitergabe des Codes entstehen könnten.
#
# Bitte lesen Sie die vollständigen Lizenzbedingungen der MIT-Lizenz, um sich mit Ihren Rechten und Pflichten vertraut zu machen.

# Français | Peharge: Ce code source est publié sous la licence MIT.
#
# Droits d'utilisation:
# Le code source peut être copié, édité et adapté aux besoins individuels.
# Les utilisateurs sont autorisés à utiliser ce code dans leurs propres projets, à des fins privées et commerciales.
# Il est cependant recommandé d'adapter le code uniquement si vous avez des connaissances suffisantes en programmation,
# car les modifications pourraient provoquer des erreurs involontaires ou des risques de sécurité.
#
# Dépendances et frameworks supplémentaires:
# Le code est basé sur l'utilisation de différents frameworks et exécute des fichiers supplémentaires.
# Certains de ces fichiers peuvent installer automatiquement des dépendances supplémentaires requises pour la fonctionnalité.
# Il est fortement recommandé d'effectuer l'installation et la configuration dans un environnement isolé (par exemple un environnement virtuel),
# pour éviter d'éventuels conflits avec les installations de logiciels existantes.
#
# Clause de non-responsabilité:
# L'utilisation du code est entièrement à vos propres risques.
# Peharge n'assume aucune responsabilité pour tout dommage, perte de données, erreurs système ou autres problèmes,
# pouvant découler directement ou indirectement de l'utilisation, de la modification ou de la diffusion du code.
#
# Veuillez lire l'intégralité des termes et conditions de la licence MIT pour vous familiariser avec vos droits et responsabilités.

import os
import numpy as np
import nibabel as nib
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import torch
from monai.transforms import Compose, LoadImaged, AddChanneld, Spacingd, ScaleIntensityd, EnsureTyped
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
import matplotlib.pyplot as plt

# Path for saving uploaded files
UPLOAD_DIR = "uploads/"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Load pre-trained MONAI Transformer model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SwinUNETR(
    img_size=(96, 96, 96),
    in_channels=1,
    out_channels=1,
    feature_size=48,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    dropout_path_rate=0.0,
).to(DEVICE)
model.load_state_dict(torch.load("path_to_pretrained_model.pth"))
model.eval()

# Define preprocessing pipeline
preprocessing = Compose([
    LoadImaged(keys=["image"]),
    AddChanneld(keys=["image"]),
    Spacingd(keys=["image"], pixdim=(1.5, 1.5, 1.5), mode=("bilinear")),
    ScaleIntensityd(keys=["image"]),
    EnsureTyped(keys=["image"]),
])

# Django views
@csrf_exempt
def upload_image(request):
    if request.method == 'POST' and request.FILES['nii_file']:
        # Save uploaded file
        nii_file = request.FILES['nii_file']
        fs = FileSystemStorage(location=UPLOAD_DIR)
        file_path = fs.save(nii_file.name, nii_file)
        full_path = os.path.join(UPLOAD_DIR, file_path)

        # Run model inference
        segmentation_path = run_segmentation(full_path)

        # Return path to segmented overlay image
        return JsonResponse({"segmented_image": segmentation_path})

    return render(request, "upload.html")

def run_segmentation(image_path):
    # Load NIfTI file
    data = {"image": image_path}
    data = preprocessing(data)
    image = data["image"].unsqueeze(0).to(DEVICE)

    # Perform inference
    with torch.no_grad():
        outputs = sliding_window_inference(image, roi_size=(96, 96, 96), sw_batch_size=4, predictor=model)
        outputs = torch.sigmoid(outputs).cpu().numpy()[0, 0]

    # Overlay segmentation on original slice
    nii = nib.load(image_path)
    img_data = nii.get_fdata()
    overlay_path = create_overlay(img_data, outputs, image_path)

    return overlay_path

def create_overlay(image_data, segmentation_data, original_path):
    # Select a slice for visualization (e.g., middle slice)
    slice_index = image_data.shape[2] // 2
    original_slice = image_data[:, :, slice_index]
    segmentation_slice = segmentation_data[:, :, slice_index]

    # Plot and save overlay
    plt.figure(figsize=(10, 10))
    plt.imshow(original_slice, cmap="gray")
    plt.imshow(segmentation_slice, cmap="jet", alpha=0.5)
    plt.axis("off")

    overlay_path = original_path.replace(".nii.gz", "_overlay.png")
    plt.savefig(overlay_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    return overlay_path