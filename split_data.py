import os
import random
import shutil



BASE_DIR = 'my_traffic_data'


SRC_IMG_DIR = os.path.join(BASE_DIR, 'images', 'train')
SRC_LBL_DIR = os.path.join(BASE_DIR, 'labels', 'train')


VAL_IMG_DIR = os.path.join(BASE_DIR, 'images', 'val')
VAL_LBL_DIR = os.path.join(BASE_DIR, 'labels', 'val')

TEST_IMG_DIR = os.path.join(BASE_DIR, 'images', 'test')
TEST_LBL_DIR = os.path.join(BASE_DIR, 'labels', 'test')


def auto_split():

    if not os.path.exists(SRC_IMG_DIR):
        print(f"HATA: {SRC_IMG_DIR} bulunamadı!")
        return


    os.makedirs(VAL_IMG_DIR, exist_ok=True)
    os.makedirs(VAL_LBL_DIR, exist_ok=True)
    os.makedirs(TEST_IMG_DIR, exist_ok=True)
    os.makedirs(TEST_LBL_DIR, exist_ok=True)


    images = [f for f in os.listdir(SRC_IMG_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
    total_count = len(images)

    if total_count == 0:
        print("HATA: Train klasöründe hiç resim yok! Resimlerin orada olduğundan emin ol.")
        return

    print(f"Toplam Resim Sayısı: {total_count}")


    random.shuffle(images)


    test_count = int(total_count * 0.15)
    val_count = int(total_count * 0.15)


    train_count = total_count - test_count - val_count

    print(f"Planlanan Dağılım -> Train: {train_count}, Val: {val_count}, Test: {test_count}")




    test_files = images[:test_count]
    remaining_files = images[test_count:]

    for filename in test_files:
        move_file_pair(filename, SRC_IMG_DIR, SRC_LBL_DIR, TEST_IMG_DIR, TEST_LBL_DIR)


    val_files = remaining_files[:val_count]

    for filename in val_files:
        move_file_pair(filename, SRC_IMG_DIR, SRC_LBL_DIR, VAL_IMG_DIR, VAL_LBL_DIR)

    print("\n✅ İşlem Tamamlandı!")
    print(f"Son Durum: Train klasöründe {len(os.listdir(SRC_IMG_DIR))} resim kaldı.")


def move_file_pair(filename, src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir):

    shutil.move(os.path.join(src_img_dir, filename), os.path.join(dst_img_dir, filename))


    label_name = os.path.splitext(filename)[0] + '.txt'
    src_label = os.path.join(src_lbl_dir, label_name)
    dst_label = os.path.join(dst_lbl_dir, label_name)

    if os.path.exists(src_label):
        shutil.move(src_label, dst_label)
    else:
        print(f"UYARI: {filename} için etiket dosyası bulunamadı!")


if __name__ == "__main__":
    auto_split()