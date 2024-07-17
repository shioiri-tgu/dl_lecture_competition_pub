import random
import numpy as np
import torch
import re
from torchvision import transforms


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def preprocess_question(question: str) -> str:
    # 全ての単語を小文字に変換
    question = question.lower()
    # 冠詞の削除
    question = re.sub(r'\b(a|an|the)\b', '', question)
    # 特殊文字の削除（必要に応じて）
    question = re.sub(r'[^\w\s]', '', question)
    # 余分な空白の削除
    question = re.sub(r'\s+', ' ', question).strip()
    return question

def get_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224),  # ランダムに切り取ってリサイズ
        transforms.RandomHorizontalFlip(),  # ランダムに左右反転
        transforms.RandomRotation(degrees=(-15, 15)),  # ランダムに回転
        transforms.RandomCrop(32, padding=(4, 4, 4, 4)),# ランダムにクロップ  
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 色調整
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正規化
    ])

def get_test_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正規化
    ])
