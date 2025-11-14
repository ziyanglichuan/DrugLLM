import os
import shutil

ASSAY_IDS = [
    "CHEMBL1613983","CHEMBL1738500",
    "CHEMBL1614183","CHEMBL1963888","CHEMBL4296185","CHEMBL4296190",
    "CHEMBL1613886","CHEMBL1614481","CHEMBL1963722","CHEMBL1963723",
    "CHEMBL1963727","CHEMBL1963788","CHEMBL1963790","CHEMBL1963807",
    "CHEMBL1963814","CHEMBL1963835","CHEMBL1964107","CHEMBL1964119"
]

def clean_directory(target_dir):
    files = os.listdir(target_dir)

    for f in files:
        full_path = os.path.join(target_dir, f)

        # 跳过子目录
        if os.path.isdir(full_path):
            continue
        
        # 判断是否保留
        keep = any(assay_id in f for assay_id in ASSAY_IDS)

        if not keep:
            print("Deleting:", full_path)
            os.remove(full_path)

    print("Done.")

if __name__ == "__main__":
    target_dir = "./sub_target_data"   # 换成你的目录
    clean_directory(target_dir)
