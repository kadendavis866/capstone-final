import math
import random
import shutil
import statistics
from typing import List, Dict

import pandas as pd
from pathlib import Path

import classify


def stage_one(src_dir: str, meta_path: str, dst_dir: str, sub0: str, sub1: str):
    meta = pd.read_csv(meta_path, header=0)

    first_scores_all: Dict[str, float] = {}
    second_scores_all: Dict[str, float] = {}
    ages_all: Dict[str, int] = {}
    for i in meta.iterrows():
        first_scores_all[i[1]["file_path"]] = float(i[1]["face_score"])
        second_scores_all[i[1]["file_path"]] = float(i[1]["second_face_score"])
        ages_all[i[1]["file_path"]] = int(i[1]["age"])

    fs = [v for v in list(first_scores_all.values()) if not math.isinf(v) and not math.isnan(v)]
    ss = [v for v in list(second_scores_all.values()) if not math.isinf(v) and not math.isnan(v)]
    first_scores_lb = statistics.mean(fs) - statistics.stdev(fs)
    second_scores_ub = statistics.mean(ss) + statistics.stdev(ss)

    dirs: List[Path] = [f for f in Path(src_dir).glob("*") if f.is_dir()]
    for d in dirs:
        for file_full, file_name in ([(f"{d.name}/{f.name}", f.name) for f in d.rglob("*") if f.is_file()]):
            age = ages_all[file_full] if file_full in ages_all else -1
            if age < 0 or age > 100: age = -1
            status = sub0 if file_full in first_scores_all and file_full in second_scores_all else sub1
            if status == sub0:
                status = sub0 if first_scores_all[file_full] >= first_scores_lb and (
                            second_scores_all[file_full] <= second_scores_ub) or math.isnan(
                    second_scores_all[file_full]) else sub1
            dst = f"{dst_dir}/{status}/{age}/{file_name}"
            Path(dst).parent.mkdir(exist_ok=True, parents=True)
            shutil.copy2(f"{src_dir}/{file_full}", dst)


def main():
    dst_dir = "dataset"

    # IMDB-WIKI
    stage_one(src_dir="imdb_crop", meta_path="imdb.csv", dst_dir=dst_dir, sub0="a1", sub1="r1")
    stage_one(src_dir="wiki_crop", meta_path="wiki.csv", dst_dir=dst_dir, sub0="a1", sub1="r1")
    print("stage 1 filtering/sorting done")
    classify.classify_images(Path(f"{dst_dir}/r1"), Path(dst_dir), "checkpoints/best_classifier2.pt", "a1", "rejected")
    shutil.rmtree(f"{dst_dir}/r1")
    classify.classify_images(Path(f"{dst_dir}/a1"), Path(dst_dir), "checkpoints/best_classifier.pt", "a2", "rejected")
    shutil.rmtree(f"{dst_dir}/a1")

    # FGNET
    src_dir = "fgnet"
    for file_name in ([f.name for f in Path(src_dir).rglob("*") if f.is_file()]):
        age = int(file_name.split("A")[1][:2])
        dst = f"{dst_dir}/a2/{age}/{file_name}"
        Path(dst).parent.mkdir(exist_ok=True, parents=True)
        shutil.copy2(f"{src_dir}/{file_name}", dst)
    print("FGNet added")

    # AgeDB
    src_dir = "AgeDB"
    for file_name in ([f.name for f in Path(src_dir).rglob("*") if f.is_file()]):
        age = int(file_name.split("_")[2])
        dst = f"{dst_dir}/a2/{age}/{file_name}"
        Path(dst).parent.mkdir(exist_ok=True, parents=True)
        shutil.copy2(f"{src_dir}/{file_name}", dst)
    print("AgeDB added")

    # Morph2
    src_dir = "morph2"
    for file_name in ([f.name for f in Path(src_dir).rglob("*") if f.is_file()]):
        age = int(file_name.split(".")[0][-2:])
        dst = f"{dst_dir}/a2/{age}/{file_name}"
        Path(dst).parent.mkdir(exist_ok=True, parents=True)
        shutil.copy2(f"{src_dir}/{file_name}", dst)
    print("Morph2 added")

    # UTKFace
    src_dir = "UTKFace"
    for file_name in ([f.name for f in Path(src_dir).rglob("*") if f.is_file()]):
        age = int(file_name.split("_")[0])
        dst = f"{dst_dir}/a2/{age}/{file_name}"
        Path(dst).parent.mkdir(exist_ok=True, parents=True)
        shutil.copy2(f"{src_dir}/{file_name}", dst)
    print("UTKFace added")

    print("Applying final filtering")
    classify.classify_images(Path(f"{dst_dir}/a2"), Path(dst_dir), "checkpoints/best_classifier3.pt", "rejected", "accepted")
    shutil.rmtree(f"{dst_dir}/a2")

    # shuffle and rename
    for age in ([f.name for f in Path(f"{dst_dir}/accepted").rglob("*") if f.is_dir()]):
        dir_name = f"{dst_dir}/accepted/{age}"
        files = [f for f in Path(dir_name).rglob("*") if f.is_file()]
        random.shuffle(files)
        Path(f"{dst_dir}/shuffled/{age}").mkdir(exist_ok=True, parents=True)
        for i, file in enumerate(files):
            shutil.copy2(f"{dir_name}/{file.name}", f"{dst_dir}/shuffled/{age}/{i}{file.suffix}")



if __name__ == "__main__":
    main()
