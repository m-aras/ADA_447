# Gereken kütüphaneler
from fastai.vision.all import *
from pathlib import Path
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np
def print_metrics(learn):
    """
    Epoch başına loss ve accuracy yazdırır.
    """
    print(f"\nEpoch  | train_loss | valid_loss | accuracy")
    for i, row in enumerate(learn.recorder.values):
        train_loss, valid_loss, acc = row[:3]
        print(f"{i:<7} | {train_loss:<10.4f} | {valid_loss:<10.4f} | {acc:.4f}")

# A.1. Veri yolunu tanımla
path = Path('./chest_xray')

# A.1.1. Veri yapısını kontrol et
print("Train klasörleri:", os.listdir(path/'train'))
print("Test klasörleri:", os.listdir(path/'test'))

# A.1.2. Veri ayırma: train klasöründeki verileri random şekilde bölelim (validation ve training)
# Train ve Test klasöründeki tüm dosyaları al
train_path = path/'train'
test_path = path/'test'
train_files = get_image_files(train_path)
test_files = get_image_files(test_path)

# Tüm dosyaları birleştir
all_files = train_files + test_files

# Her dosyanın label'ını al
from collections import Counter
labels = [parent_label(f) for f in all_files]
label_counts = Counter(labels)

print("\nToplam her sınıftan örnek sayısı (train + test):")
for label, count in label_counts.items():
    print(f"{label}: {count}")

# `RandomSplitter` kullanarak veriyi eğitim ve doğrulama setlerine ayıralım
train_path = path/'train'
files = get_image_files(train_path)

# A.2. DataBlock oluştur
# A.2. DataBlock oluştur
dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),                 # Giriş: Resim, Çıkış: Etiket
    get_items=get_image_files,                         # get item  yerine  get image
    splitter=RandomSplitter(valid_pct=0.2, seed=42),   # Eğitim/Doğrulama ayırımı
    get_y=parent_label,                                # Etiket: klasör ismi
    item_tfms=Resize(256),                             # Presize: 256x256
    batch_tfms=aug_transforms(size=224)                # Augmentasyon + 224 boyutlandırma
)

# DataLoader oluştur
dls = dblock.dataloaders(train_path, bs=32, num_workers=0)

# A.3.1. Görsel batch göster
dls.show_batch(max_n=6, figsize=(7,8))

# A.3.2. Etiketleri yazdır
print(f"\nClass names: {dls.vocab}")

# A.3.3. DataLoader özeti
print(f"DataLoader Summary:")
dblock.summary(path/'train')
print(f"Number of batches: {len(dls.train)}")
dblock.summary(path)


# A.4. Basit model oluştur
learn = vision_learner(dls, resnet50 , metrics=accuracy)

# B.1. Learning rate finder
# ResNet50 ile yeni model oluştur
learn = vision_learner(dls, resnet50, metrics=accuracy)
learn.to_fp16()  # B.6: Mixed precision kullan

# B.1. Learning Rate Finder 
lr_find_result = learn.lr_find()

# En düşük öğrenme oranı (valley) B.1.2 
low_lr = lr_find_result.valley

# Düşük lr'nin 10 katı olarak yüksek öğrenme oranı belirle B.1.1 
high_lr = low_lr * 10

# Sonuçları yazdır
print(f"Learning Rate Finder sonucu:")
print(f" - Düşük öğrenme oranı (valley): {low_lr}")
print(f" - Yüksek öğrenme oranı (valley x10): {high_lr}")


lrs = learn.recorder.lrs
losses = learn.recorder.losses

# Plot
plt.figure(figsize=(8,5))
plt.plot(lrs, losses)
plt.xscale("log")
plt.xlabel("Learning Rate")
plt.ylabel("Loss")
plt.title("Learning Rate Finder (Valley & High LR işaretli)")

# low_lr noktası (valley)
plt.scatter(low_lr, min(losses), color='green', label=f"Low LR (valley): {low_lr:.2e}")

# high_lr noktası
plt.scatter(high_lr, min(losses), color='orange', label=f"High LR: {high_lr:.2e}")

plt.legend()
plt.grid(True)
plt.show()
# B.2. Sadece head eğit
learn.freeze()
print("\n--- ResNet50 (Frozen: Only head trained) ---")
learn.fit_one_cycle(3)
print("\nMetrics (Frozen - ResNet50):")
print_metrics(learn)

# B.3. Tüm modeli eğit (unfreeze)
learn.unfreeze()
print("\n--- ResNet50 (Unfrozen: Entire model trained) ---")
learn.fit_one_cycle(3, lr_max=slice(1e-6,1e-3))
print("\nMetrics (Unfrozen - ResNet50):")
print_metrics(learn)

# B.6. Modeli Mixed Precision moduna al (FP16)
if torch.cuda.is_available():
    learn.to_fp16()
else:
    print("CUDA bulunamadı, fp16 devre dışı.")

# B.4. Discriminative Learning Rates ile eğit
learn.fit_one_cycle(3, lr_max=slice(low_lr, high_lr))

print("\nMetrics (Unfrozen):")
print_metrics(learn)

# A.4.2 & A.4.3. Modeli yorumla ve confusion matrix göster
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
plt.show()
interp.plot_top_losses(6, nrows=2)
interp.most_confused(min_val=2)
plt.show()


# Tahminleri ve gerçek etiketleri al
preds, targs = learn.get_preds()

# Sınıf sayısı
n_classes = len(dls.vocab)

if n_classes == 2:
    # İkili sınıflandırma için ROC ve AUC
    fpr, tpr, thresholds = roc_curve(targs, preds[:,1])  # 1. sınıfın olasılığı
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7,5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

else:
    # Çok sınıflı sınıflandırma için ROC ve AUC
    y_true = label_binarize(targs, classes=range(n_classes))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure(figsize=(8,6))
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {dls.vocab[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Multi-class')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
