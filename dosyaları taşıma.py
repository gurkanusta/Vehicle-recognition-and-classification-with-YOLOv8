import os
import shutil



SRC_IMAGES = 'new_images_to_label/'

SRC_LABELS = 'auto_label_results/labels/labels/'


DST_IMAGES = 'my_traffic_data/images/train/'
DST_LABELS = 'my_traffic_data/labels/train/'


def merge_files():
    print("📦 Dosya taşıma işlemi başlıyor...")


    if os.path.exists(SRC_IMAGES):
        images = [f for f in os.listdir(SRC_IMAGES) if f.endswith(('.jpg', '.png', '.jpeg'))]
        print(f"➡️  {len(images)} adet resim taşınıyor...")

        for img in images:
            try:
                shutil.move(os.path.join(SRC_IMAGES, img), os.path.join(DST_IMAGES, img))
            except Exception as e:
                print(f"⚠️ Hata (Resim): {img} taşınamadı. {e}")
    else:
        print("Kaynak resim klasörü bulunamadı!")


    if os.path.exists(SRC_LABELS):
        labels = [f for f in os.listdir(SRC_LABELS) if f.endswith('.txt')]
        print(f"➡️  {len(labels)} adet etiket dosyası taşınıyor...")

        for lbl in labels:
            try:
                shutil.move(os.path.join(SRC_LABELS, lbl), os.path.join(DST_LABELS, lbl))
            except Exception as e:
                print(f"⚠️ Hata (Etiket): {lbl} taşınamadı. {e}")
    else:
        print(f"❌ Kaynak etiket klasörü bulunamadı: {SRC_LABELS}")

        alt_src = 'auto_label_results/labels/'
        if os.path.exists(alt_src):
            print(f"ℹ️ Alternatif yol kontrol ediliyor: {alt_src}")

            labels = [f for f in os.listdir(alt_src) if f.endswith('.txt')]
            for lbl in labels:
                shutil.move(os.path.join(alt_src, lbl), os.path.join(DST_LABELS, lbl))

    print("\n✅ TÜM DOSYALAR BİRLEŞTİRİLDİ!")
    print(f"Resimler: {DST_IMAGES}")
    print(f"Etiketler: {DST_LABELS}")
    print("Şimdi LabelImg ile 'my_traffic_data/images/train' klasörünü açıp kontrol edebilirsin.")


if __name__ == '__main__':
    merge_files()