# pytorch-cheat-sheet
- https://pytorch.org/get-started/locally/
- PyTorch ist eine mächtige Open-Source-Maschinenlernbibliothek, die von Facebook AI entwickelt wurde. Sie wird besonders für die Implementierung von Deep-Learning-Modellen genutzt. PyTorch bietet dynamisches Rechendefinieren, was bedeutet, dass du Modelle erstellen und ändern kannst, während sie ausgeführt werden.



<br><br>
<br><br>

## Install

<br><br>

### Ubuntu 24.04

<br>

#### pip
```shell
# pip

# Virtuelle Umgebung löschen, falls sie existiert
rm -rf venv

# Virtuelle Umgebung erstellen
python3 -m venv venv
source venv/bin/activate

# CUDA-Version erkennen
CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
echo "CUDA-Version: $CUDA_VERSION"

# Installationsbefehl für PyTorch basierend auf der CUDA-Version
if [[ $CUDA_VERSION == 12.* ]]; then
    pip install torch --extra-index-url https://download.pytorch.org/whl/cu124
elif [[ $CUDA_VERSION == 11.* ]]; then
    pip install torch --extra-index-url https://download.pytorch.org/whl/cu117
elif [[ $CUDA_VERSION == 10.* ]]; then
    pip install torch --extra-index-url https://download.pytorch.org/whl/cu102
else
    echo "CUDA-Version nicht unterstützt oder nicht gefunden."
    exit 1
fi

# anaconda
conda install pytorch torchvision
conda install pytorch::torchaudio
```

<br>

#### anaconda
```shell
conda install pytorch torchvision
conda install pytorch::torchaudio
```











<br><br>
<br><br>
---
<br><br>
<br><br>

# Device

## Check if CPU or GPU
```python
import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

# Überprüfe, ob die GPU verfügbar ist
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    print("GPU wird verwendet:", torch.cuda.get_device_name(0))
else:
    print("GPU nicht verfügbar, CPU wird verwendet.")

# Kosmos-2 Model und Processor laden
kosmos_path = "/home/t33n/Projects/ai/resources/transformers/kosmos-2-patch14-224"
model = AutoModelForVision2Seq.from_pretrained(kosmos_path).to(device)  # Verschiebe das Modell auf die GPU
processor = AutoProcessor.from_pretrained(kosmos_path)

def generate_caption_for_image(image_path):
    # Bild öffnen
    image = Image.open(image_path)
    
    # Eingabeprompt vorbereiten
    prompt = "<grounding>An image of"
    
    # Bild und Text durch Processor verarbeiten
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    
    # Verschiebe Eingaben auf die GPU
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Bildbeschreibung generieren
    generated_ids = model.generate(
        pixel_values=inputs["pixel_values"],
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_embeds=None,
        image_embeds_position_mask=inputs["image_embeds_position_mask"],
        use_cache=True,
        max_new_tokens=128,
    )
    
    # Text dekodieren und generieren
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    processed_text, _ = processor.post_process_generation(generated_text)
    
    return processed_text

def process_images_in_directory(directory_path):
    # Alle Bilddateien im Verzeichnis durchsuchen
    for filename in os.listdir(directory_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(directory_path, filename)
            caption = generate_caption_for_image(image_path)
            
            # .txt Datei mit Bildnamen erstellen und Caption speichern
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            txt_path = os.path.join(directory_path, txt_filename)

            with open(txt_path, "w") as txt_file:
                txt_file.write(caption)
                
            print(f"Caption für '{filename}' erstellt und in '{txt_filename}' gespeichert.")

# Beispiel für die Nutzung: Verzeichnispfad angeben
directory_path = "./imgs"
process_images_in_directory(directory_path)
```


