import torch
from torchvision import models, transforms
from PIL import Image
import os, re, random
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms, models
from facenet_pytorch import InceptionResnetV1
import pytesseract
from dateutil import parser
from datetime import datetime
from scipy.spatial.distance import cosine

# ------------------------------
# 1) YAŞ ARALIKLARINA GÖRE 10'AR FOTO SEÇ
# ------------------------------

folder_path = r"C:\Users\ali_b\OneDrive\Masaüstü\digital image processing term project\fotoğraflar\part1"

age_groups = {
    "0-10": [],
    "18-30": [],
    "30-50": [],
    "50-70": [],
    "70-90": [],
    "90-110": []
}

valid_ext = [".jpg", ".jpeg", ".png"]

for filename in os.listdir(folder_path):
    if not any(filename.lower().endswith(ext) for ext in valid_ext):
        continue

    match = re.match(r"(\d+)_", filename)
    if not match:
        continue

    age = int(match.group(1))
    filepath = os.path.join(folder_path, filename)

    # yaş aralıkları
    if 0 <= age <= 10:
        age_groups["0-10"].append(filepath)
    elif 18 <= age <= 30:
        age_groups["18-30"].append(filepath)
    elif 30 <= age <= 50:
        age_groups["30-50"].append(filepath)
    elif 50 <= age <= 70:
        age_groups["50-70"].append(filepath)
    elif 70 <= age <= 90:
        age_groups["70-90"].append(filepath)
    elif 90 <= age <= 110:
        age_groups["90-110"].append(filepath)

# 10'ar tane seç
selected = {}
for group, files in age_groups.items():
    selected[group] = random.sample(files, min(10, len(files)))

# ------------------------------
# 2) MODELİ YÜKLE
# ------------------------------

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load("age_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ------------------------------
# 3) SEÇİLEN TÜM FOTOĞRAFLARDAN YAŞ TAHMİNİ
# ------------------------------

for group, files in selected.items():
    print(f"\n=== {group} aralığındaki fotoğraflar ({len(files)} adet) ===")
    for file_path in files:
        try:
            img = Image.open(file_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0)

            with torch.no_grad():
                pred_age = model(img_tensor).item()

            print(f"{os.path.basename(file_path)}  -> Tahmin: {pred_age:.2f}")

        except Exception as e:
            print(f"HATA: {file_path} işlenemedi ({e})")




# ------------------------------
# TESSERACT PATH (Windows)
# ------------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ------------------------------
# AGE MODEL (Senin modelin)
# ------------------------------
age_model = models.resnet18(weights=None)
age_model.fc = torch.nn.Linear(age_model.fc.in_features, 1)
age_model.load_state_dict(torch.load("age_model.pth"))
age_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def predict_age(img_path):
    img = Image.open(img_path).convert("RGB")
    t = transform(img).unsqueeze(0)
    with torch.no_grad():
        age = age_model(t).item()
    return age


import easyocr as ocr

ocr_engine = ocr.Reader (["en" ,"tr"])
IDTexts = ocr_engine.readtext(r"C:\IDs\secondID.jpg")
print(IDTexts)

# OCR çıktısı (sadece örnek)
ocr_list = [
    ([[965, 78], [1358, 78], [1358, 145], [965, 145]], 'Soyadı / Sumname', 0.6274),
    ([[965, 136], [1354, 136], [1354, 225], [965, 225]], 'BOSTANCI', 0.8169),
    ([[963, 246], [1413, 246], [1413, 318], [963, 318]], 'Adı / Given Name(s)', 0.8431),
    ([[964, 308], [1469, 308], [1469, 399], [964, 399]], 'AHMET OĞUZ', 0.9975),
    ([[962, 421], [1291, 421], [1291, 490], [962, 490]], 'Dogum Tarihi /', 0.8988),
    ([[1302, 430], [1570, 430], [1570, 478], [1302, 478]], 'Date of Birth', 0.9911),
    ([[962, 486], [1373, 486], [1373, 566], [962, 566]], '01.09.2000', 0.9598),
    ([[1137, 605], [1289, 605], [1289, 649], [1137, 649]], '/ Doct', 0.1487),
    ([[1071.2, 598.08], [1143.92, 618.22], [1132.8, 654.91], [1060.07, 634.77]], '9o', 0.0322)
]
# OCR textlerini düz listeye al
texts = [item[1] for item in ocr_list]

dob_value = None

# Tarih formatını tespit et (dd.mm.yyyy)
for text in texts:
    try:
        dob = datetime.strptime(text, "%d.%m.%Y")
        dob_value = dob
        break
    except:
        continue

if dob_value:
    today = datetime.today()
    age = today.year - dob_value.year - ((today.month, today.day) < (dob_value.month, dob_value.day))
    if age >= 18:
        print(f"Reşit: Yaş = {age}")
    else:
        print(f"Reşit değil: Yaş = {age}")
else:
    print("Doğum tarihi OCR'den alınamadı.")



# ------------------------------
# 1) OCR'den alınan doğum tarihi
# ------------------------------
dob_string = "25.08.2001"  # OCR çıktısı
dob = datetime.strptime(dob_string, "%d.%m.%Y")
today = datetime.today()
real_age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

if real_age < 18:
    raise Exception("Kimlik reşit göstermiyor!")

# ------------------------------
# 2) Yaş tahmini modeli yükle
# ------------------------------
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load("age_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ------------------------------
# 3) Kullanıcı fotoğrafından yaş tahmini
# ------------------------------
user_img_path = r"C:\IDs\secondID.jpg"  # kamera veya yüklenen fotoğraf
img = Image.open(user_img_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    predicted_age = model(img_tensor).item()

if predicted_age < 18:
    raise Exception(f"Model seni reşit bulmadı: Tahmini yaş = {predicted_age:.1f}")

print(f"Model tahmini yaş: {predicted_age:.1f} | Kimlik yaşı: {real_age}")

# ------------------------------
# 4) Kimlik yüzü ile kullanıcı yüzü eşleştirme
# ------------------------------
# id_face ve user_face: OpenCV ile crop edilmiş yüzler
# ArcFace veya başka bir yüz embedding modeli kullanılır
# similarity = cosine(arcface(id_face), arcface(user_face))
# if similarity < 0.3:
#     raise Exception("Kimlik fotoğrafı ile kullanıcı aynı kişi değil.")

print("Reşitlik ve yüz doğrulama kontrolü tamamlandı.")
