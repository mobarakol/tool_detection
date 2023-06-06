from glob import glob
import shutil, os
from shutil import copyfile

dest_dir = 'dataset/processed/'
data_dir = ['dataset/instruments18/seq_']
img_dir = ['/left_frames/']
label_dir = ['/xml/']

train_seq = [2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15]
val_seq = [1, 5, 16]
all_seqs = train_seq + val_seq
dset = [0]

def move_files_to_folder(list_of_files, destination_root):
    for f in list_of_files:
        seq_n = f.split("/")[-3]
        img_n = f.split("/")[-1]
        new_n = seq_n + "_" + img_n
        destination_file = destination_root + "/" + new_n

        if not os.path.exists(destination_root):
            os.makedirs(destination_root)

        try:
            copyfile(f, destination_file)
        except:
            print(f)
            assert False

# for item in val_seq:
#     seq_num = data_dir[0] + str(item) + "/"
#     imgs = glob(seq_num + "left_frames/*.png")
#     labels = glob(seq_num + "xml/*.txt")
#     move_files_to_folder(imgs, dest_dir + 'images/val')
#     move_files_to_folder(labels, dest_dir + 'labels/val')

# for item in train_seq:
#     seq_num = data_dir[0] + str(item) + "/"
#     imgs = glob(seq_num + "left_frames/*.png")
#     labels = glob(seq_num + "xml/*.txt")
#     move_files_to_folder(imgs, dest_dir + 'images/train')
#     move_files_to_folder(labels, dest_dir + 'labels/train')

# CUDA_VISIBLE_DEVICES=0 python3 train.py --img 640 --cfg yolov5s.yaml --hyp hyp.scratch.yaml --batch 16 --epochs 100 --data endovis18.yaml --weights yolov5s.pt --workers 8 --name yolo_endo18    

for item in train_seq:
    seq_num = data_dir[0] + str(item) + "/"
    labels = glob(seq_num + "xml/*.xml")
    move_files_to_folder(labels, dest_dir + 'labels_xml/train')

for item in val_seq:
    seq_num = data_dir[0] + str(item) + "/"
    labels = glob(seq_num + "xml/*.xml")
    move_files_to_folder(labels, dest_dir + 'labels_xml/val')