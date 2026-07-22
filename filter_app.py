
import os
import io
import threading
import time
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageEnhance, ImageFilter
import customtkinter as ctk
from tkinter import filedialog, messagebox
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
                                                              
DARK_BG       = "#0D0D0F"
CARD_BG       = "#1A1A1F"
ACCENT_PURPLE = "#8B5CF6"
ACCENT_PINK   = "#EC4899"
ACCENT_CYAN   = "#06B6D4"
TEXT_PRIMARY  = "#F1F5F9"
TEXT_MUTED    = "#64748B"
SUCCESS       = "#10B981"
BORDER        = "#2D2D35"
                                                                  
FILTERS = {
    "Vintage":       {"emoji": "",  "desc": "Warm sepia tones with faded edges"},
    "B&W":           {"emoji": "",  "desc": "Timeless black-and-white conversion"},
    "Warm Tone":     {"emoji": "",  "desc": "Golden warm hues for a cozy feel"},
    "Cool Tone":     {"emoji": "",  "desc": "Crisp blue-tinted atmospheric look"},
    "Bright Boost":  {"emoji": "",  "desc": "Vivid brightness & punchy contrast"},
    "Sunset Glow":   {"emoji": "",  "desc": "Orange-gold sunset magic"},
    "Aesthetic Pink":{"emoji": "",  "desc": "Dreamy pastel pink overlay"},
    "Matte":         {"emoji": "",  "desc": "Flat matte finish with lifted blacks"},
    "Neon Glow":     {"emoji": "",  "desc": "Electric neon-lit vibrant pop"},
    "Cinematic":     {"emoji": "",  "desc": "Widescreen teal-orange cinema grade"},
    "Soft Focus":    {"emoji": "",  "desc": "Gentle blur for a dreamy portrait"},
    "Faded":         {"emoji": "",  "desc": "Washed-out retro faded look"},
    "Dramatic":      {"emoji": "",  "desc": "High-contrast intense dark mood"},
    "Aqua":          {"emoji": "",  "desc": "Teal-green underwater freshness"},
    "Cross Process": {"emoji": "",  "desc": "Experimental cross-processed colors"},
    "Cyberpunk":     {"emoji": "",  "desc": "Dark purple-cyan neon dystopia"},
    "Golden Hour":   {"emoji": "",  "desc": "Magic hour warm golden sunlight"},
    "Moonlight":     {"emoji": "",  "desc": "Cold blue ethereal night glow"},
    "Polaroid":      {"emoji": "",  "desc": "Instant camera warmth & borders"},
    "Lomo":          {"emoji": "",  "desc": "High-sat vignette toy camera"},
    "Velvet":        {"emoji": "",  "desc": "Smooth rich dark luxe tones"},
    "Arctic":        {"emoji": "",  "desc": "Icy pale frozen landscape"},
    "Cherry Blossom":{"emoji": "",  "desc": "Soft pink-purple spring bloom"},
    "Emerald":       {"emoji": "",  "desc": "Rich deep green jewel tones"},
    "Lavender":      {"emoji": "",  "desc": "Purple pastel soft dream wash"},
    "Burnt Orange":  {"emoji": "",  "desc": "Deep terracotta warm earth"},
    "Film Noir":     {"emoji": "",  "desc": "Gritty high-contrast detective B&W"},
    "Retro 70s":     {"emoji": "",  "desc": "Yellow-brown groovy flashback"},
    "Vaporwave":     {"emoji": "",  "desc": "Pink-purple-cyan retro-future"},
    "Desert Sand":   {"emoji": "",  "desc": "Sandy warm muted earth palette"},
    "Ocean Breeze":  {"emoji": "",  "desc": "Cool refreshing coastal blues"},
    "Autumn":        {"emoji": "",  "desc": "Warm red-orange harvest glow"},
    "Infrared":      {"emoji": "",  "desc": "False-color infrared vision"},
    "Noir Gold":     {"emoji": "",  "desc": "Dark moody with gold accents"},
    "Pastel":        {"emoji": "",  "desc": "Soft airy washed pastel candy"},
    "HDR Effect":    {"emoji": "",  "desc": "Hyper-detailed clarity boost"},
    "Bleach Bypass": {"emoji": "",  "desc": "Silver-retained desaturated punch"},
    "Rose Gold":     {"emoji": "",  "desc": "Luxe pink-gold metallic warmth"},
    "Midnight":      {"emoji": "",  "desc": "Deep dark blue starry mood"},
    "Candy Pop":     {"emoji": "",  "desc": "Super vivid candy explosion"},
    "Sepia Classic": {"emoji": "",  "desc": "Traditional old-photo sepia"},
    "Chrome":        {"emoji": "",  "desc": "Metallic silver futuristic sheen"},
    "Foggy Morning": {"emoji": "",  "desc": "Soft hazy gentle morning light"},
    "Tropical":      {"emoji": "",  "desc": "Vibrant lush tropical palette"},
    "Electric Blue": {"emoji": "",  "desc": "Strong vivid blue intensity"},
    "Copper":        {"emoji": "",  "desc": "Warm copper metallic glow"},
    "Dream Glow":    {"emoji": "",  "desc": "Soft bloom ethereal light wrap"},
    "Duotone Purple":{"emoji": "",  "desc": "Two-tone purple shadow map"},
    "Duotone Teal":  {"emoji": "",  "desc": "Two-tone teal gradient map"},
    "Noir Crimson":  {"emoji": "",  "desc": "Dark noir with red accent bleed"},
}
                                                              
def extract_features(pil_image: Image.Image) -> dict:
    img_rgb = np.array(pil_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    saturation = float(np.mean(hsv[:, :, 1]))
    dominant_hue = float(np.median(hsv[:, :, 0]))                    
    r, g, b = img_rgb[:,:,0].astype(float), img_rgb[:,:,1].astype(float), img_rgb[:,:,2].astype(float)
    rg = np.abs(r - g)
    yb = np.abs(0.5*(r + g) - b)
    colorfulness = float(np.sqrt(np.mean(rg)**2 + np.mean(yb)**2) +
                         0.3 * np.sqrt(np.std(rg)**2 + np.std(yb)**2))
    pixels = img_rgb.reshape(-1, 3).astype(np.float32)
    sample = pixels[np.random.choice(len(pixels), min(2000, len(pixels)), replace=False)]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers = cv2.kmeans(sample, 3, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
    dominant_colors = [tuple(int(c) for c in col) for col in centers.astype(int)]
    warm_mask = ((hsv[:,:,0] <= 30) | (hsv[:,:,0] >= 150)) & (hsv[:,:,1] > 50)
    warm_ratio = float(np.sum(warm_mask) / warm_mask.size)
    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    has_face = len(faces) > 0
    return {
        "brightness":      brightness,
        "contrast":        contrast,
        "saturation":      saturation,
        "dominant_hue":    dominant_hue,
        "colorfulness":    colorfulness,
        "warm_ratio":      warm_ratio,
        "has_face":        has_face,
        "dominant_colors": dominant_colors,
    }
def get_dominant_color_hex(dominant_colors: list) -> str:
    best = max(dominant_colors, key=lambda c: (c[0]-128)**2 + (c[1]-128)**2 + (c[2]-128)**2)
    return "#{:02X}{:02X}{:02X}".format(*best)
                                    
                                                                            
FILTER_NAMES = list(FILTERS.keys())                
FILTER_PROFILES = {
    "Vintage":       [120, 40, 80,  0.60, 30],
    "B&W":           [140, 58, 20,  0.20, 10],
    "Warm Tone":     [170, 45, 100, 0.70, 45],
    "Cool Tone":     [150, 50, 90,  0.20, 40],
    "Bright Boost":  [80,  75, 100, 0.40, 55],
    "Sunset Glow":   [140, 50, 130, 0.75, 60],
    "Aesthetic Pink":[160, 35, 70,  0.35, 35],
    "Matte":         [140, 30, 60,  0.40, 22],
    "Neon Glow":     [70,  80, 150, 0.50, 80],
    "Cinematic":     [100, 65, 70,  0.55, 38],
    "Soft Focus":    [185, 25, 85,  0.45, 30],
    "Faded":         [200, 20, 40,  0.35, 15],
    "Dramatic":      [60,  90, 90,  0.40, 45],
    "Aqua":          [155, 50, 110, 0.15, 50],
    "Cross Process": [130, 70, 140, 0.50, 70],
    "Cyberpunk":     [65,  80, 130, 0.30, 75],
    "Golden Hour":   [175, 42, 105, 0.75, 50],
    "Moonlight":     [90,  45, 60,  0.15, 25],
    "Polaroid":      [155, 38, 85,  0.55, 35],
    "Lomo":          [120, 70, 140, 0.50, 65],
    "Velvet":        [95,  55, 70,  0.40, 30],
    "Arctic":        [190, 35, 40,  0.15, 18],
    "Cherry Blossom":[165, 38, 80,  0.40, 38],
    "Emerald":       [130, 50, 120, 0.30, 55],
    "Lavender":      [165, 35, 75,  0.35, 35],
    "Burnt Orange":  [125, 55, 100, 0.80, 50],
    "Film Noir":     [100, 85, 15,  0.20, 8],
    "Retro 70s":     [135, 40, 90,  0.65, 40],
    "Vaporwave":     [115, 60, 130, 0.45, 72],
    "Desert Sand":   [160, 35, 65,  0.60, 28],
    "Ocean Breeze":  [155, 45, 100, 0.15, 45],
    "Autumn":        [130, 50, 110, 0.72, 55],
    "Infrared":      [140, 55, 100, 0.40, 60],
    "Noir Gold":     [85,  70, 50,  0.55, 25],
    "Pastel":        [195, 25, 55,  0.40, 22],
    "HDR Effect":    [130, 80, 130, 0.45, 65],
    "Bleach Bypass": [125, 72, 50,  0.40, 28],
    "Rose Gold":     [155, 40, 80,  0.60, 38],
    "Midnight":      [50,  65, 70,  0.20, 30],
    "Candy Pop":     [160, 55, 150, 0.45, 80],
    "Sepia Classic": [130, 42, 60,  0.65, 28],
    "Chrome":        [160, 70, 30,  0.35, 15],
    "Foggy Morning": [185, 22, 70,  0.45, 25],
    "Tropical":      [160, 50, 130, 0.55, 65],
    "Electric Blue": [110, 60, 100, 0.15, 50],
    "Copper":        [130, 52, 85,  0.70, 40],
    "Dream Glow":    [175, 28, 80,  0.45, 30],
    "Duotone Purple":[120, 55, 90,  0.35, 45],
    "Duotone Teal":  [125, 50, 95,  0.20, 42],
    "Noir Crimson":  [80,  72, 55,  0.60, 32],
}
_train_rows = []
_train_labels = []
for idx, fname in enumerate(FILTER_NAMES):
    profile = FILTER_PROFILES.get(fname, [128, 50, 80, 0.40, 40])
    for delta in [0, 1, -1]:
        row = [
            profile[0] + delta * 8,
            profile[1] + delta * 4,
            profile[2] + delta * 6,
            max(0, min(1, profile[3] + delta * 0.05)),
            profile[4] + delta * 4,
        ]
        _train_rows.append(row)
        _train_labels.append(idx)
_TRAIN_X = np.array(_train_rows)
_TRAIN_Y = np.array(_train_labels)
_scaler = MinMaxScaler()
_TRAIN_X_SCALED = _scaler.fit_transform(_TRAIN_X)
_knn = KNeighborsClassifier(n_neighbors=3, weights="distance")
_knn.fit(_TRAIN_X_SCALED, _TRAIN_Y)
def recommend_filters(features: dict) -> list[tuple[str, float]]:
    b  = features["brightness"]
    c  = features["contrast"]
    s  = features["saturation"]
    wr = features["warm_ratio"]
    cf = features["colorfulness"]
    query = np.array([[b, c, s, wr, cf]])
    query_scaled = _scaler.transform(query)
    distances, indices = _knn.kneighbors(query_scaled, n_neighbors=min(len(_TRAIN_X), 30))
    weights = 1.0 / (distances[0] + 1e-6)
    knn_scores = np.zeros(len(FILTER_NAMES))
    for w, idx in zip(weights, indices[0]):
        knn_scores[_TRAIN_Y[idx]] += w
    knn_scores /= knn_scores.sum()
    query_vec = np.array([b, c, s, wr * 255, cf])
    rule_scores = np.zeros(len(FILTER_NAMES))
    for i, fname in enumerate(FILTER_NAMES):
        p = FILTER_PROFILES.get(fname, [128, 50, 80, 0.40, 40])
        p_vec = np.array([p[0], p[1], p[2], p[3] * 255, p[4]])
        dist = np.sqrt(np.sum((query_vec - p_vec) ** 2))
        rule_scores[i] = 1.0 / (1.0 + dist / 50.0)
    rule_scores /= rule_scores.sum() + 1e-9
    final_scores = 0.4 * knn_scores + 0.6 * rule_scores
    final_scores /= final_scores.sum()
    ranked = sorted(
        zip(FILTER_NAMES, (final_scores * 100).tolist()),
        key=lambda x: x[1], reverse=True
    )
    return ranked
                               
def apply_vintage(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    r, g, b = img.split()
    r = r.point(lambda i: min(255, int(i * 1.08)))
    b = b.point(lambda i: int(i * 0.85))
    img = Image.merge("RGB", (r, g, b))
    arr = np.array(img, dtype=np.float32)
    sr = arr[:,:,0]*0.393 + arr[:,:,1]*0.769 + arr[:,:,2]*0.189
    sg = arr[:,:,0]*0.349 + arr[:,:,1]*0.686 + arr[:,:,2]*0.168
    sb = arr[:,:,0]*0.272 + arr[:,:,1]*0.534 + arr[:,:,2]*0.131
    sepia = np.stack([sr, sg, sb], axis=2).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(sepia)
    img = ImageEnhance.Brightness(img).enhance(0.90)
    return img
def apply_bw(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(1.3)
    img = ImageEnhance.Sharpness(img).enhance(1.2)
    return img.convert("RGB")
def apply_warm(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    arr = np.array(img, dtype=np.float32)
    arr[:,:,0] = np.clip(arr[:,:,0] * 1.12, 0, 255)             
    arr[:,:,1] = np.clip(arr[:,:,1] * 1.05, 0, 255)                
    arr[:,:,2] = np.clip(arr[:,:,2] * 0.88, 0, 255)               
    img = Image.fromarray(arr.astype(np.uint8))
    img = ImageEnhance.Brightness(img).enhance(1.05)
    return img
def apply_cool(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    arr = np.array(img, dtype=np.float32)
    arr[:,:,0] = np.clip(arr[:,:,0] * 0.88, 0, 255)
    arr[:,:,1] = np.clip(arr[:,:,1] * 0.95, 0, 255)
    arr[:,:,2] = np.clip(arr[:,:,2] * 1.15, 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))
    img = ImageEnhance.Brightness(img).enhance(1.03)
    return img
def apply_bright_boost(img: Image.Image) -> Image.Image:
    img = ImageEnhance.Brightness(img.convert("RGB")).enhance(1.25)
    img = ImageEnhance.Contrast(img).enhance(1.30)
    img = ImageEnhance.Color(img).enhance(1.20)
    return img
def apply_sunset_glow(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    arr = np.array(img, dtype=np.float32)
    arr[:,:,0] = np.clip(arr[:,:,0] * 1.20, 0, 255)
    arr[:,:,1] = np.clip(arr[:,:,1] * 1.05, 0, 255)
    arr[:,:,2] = np.clip(arr[:,:,2] * 0.75, 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))
    img = ImageEnhance.Brightness(img).enhance(1.08)
    img = ImageEnhance.Color(img).enhance(1.15)
    return img
def apply_aesthetic_pink(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    arr = np.array(img, dtype=np.float32)
    arr[:,:,0] = np.clip(arr[:,:,0] * 1.10 + 15, 0, 255)
    arr[:,:,1] = np.clip(arr[:,:,1] * 0.90 + 5,  0, 255)
    arr[:,:,2] = np.clip(arr[:,:,2] * 0.92 + 10, 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))
    img = ImageEnhance.Brightness(img).enhance(1.06)
    img = ImageEnhance.Color(img).enhance(0.85)
    return img
def apply_matte(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    arr = np.array(img, dtype=np.float32)
    arr = arr * 0.85 + 30
    arr = np.clip(arr, 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))
    img = ImageEnhance.Contrast(img).enhance(0.80)
    img = ImageEnhance.Color(img).enhance(0.85)
    return img
def apply_neon_glow(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    img = ImageEnhance.Brightness(img).enhance(0.75)
    img = ImageEnhance.Color(img).enhance(2.20)
    img = ImageEnhance.Contrast(img).enhance(1.40)
    arr = np.array(img, dtype=np.float32)
    arr[:,:,2] = np.clip(arr[:,:,2] * 1.20, 0, 255)
    arr[:,:,0] = np.clip(arr[:,:,0] * 1.10, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))
def apply_cinematic(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    arr = np.array(img, dtype=np.float32)
    arr[:,:,0] = np.clip(arr[:,:,0] * 1.10, 0, 255)                             
    arr[:,:,1] = np.clip(arr[:,:,1] * 0.95, 0, 255)
    arr[:,:,2] = np.clip(arr[:,:,2] * 0.88, 0, 255)                         
    img = Image.fromarray(arr.astype(np.uint8))
    img = ImageEnhance.Contrast(img).enhance(1.25)
    img = ImageEnhance.Brightness(img).enhance(0.88)
    img = ImageEnhance.Color(img).enhance(0.90)
    return img
def apply_soft_focus(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    blurred = img.filter(ImageFilter.GaussianBlur(radius=1.8))
    img = Image.blend(img, blurred, alpha=0.55)
    img = ImageEnhance.Brightness(img).enhance(1.08)
    img = ImageEnhance.Color(img).enhance(0.90)
    return img
def apply_faded(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    arr = np.array(img, dtype=np.float32)
    arr = arr * 0.75 + 40
    arr = np.clip(arr, 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))
    img = ImageEnhance.Contrast(img).enhance(0.72)
    img = ImageEnhance.Color(img).enhance(0.80)
    img = ImageEnhance.Brightness(img).enhance(1.10)
    return img
def apply_dramatic(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    img = ImageEnhance.Brightness(img).enhance(0.72)
    img = ImageEnhance.Contrast(img).enhance(1.60)
    img = ImageEnhance.Color(img).enhance(0.75)
    arr = np.array(img, dtype=np.float32)
    arr = np.clip(arr * 0.92 - 5, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))
def apply_aqua(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    arr = np.array(img, dtype=np.float32)
    arr[:,:,0] = np.clip(arr[:,:,0] * 0.80, 0, 255)               
    arr[:,:,1] = np.clip(arr[:,:,1] * 1.10, 0, 255)                
    arr[:,:,2] = np.clip(arr[:,:,2] * 1.18, 0, 255)               
    img = Image.fromarray(arr.astype(np.uint8))
    img = ImageEnhance.Color(img).enhance(1.20)
    img = ImageEnhance.Brightness(img).enhance(1.05)
    return img
def apply_cross_process(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    arr = np.array(img, dtype=np.float32)
    r = np.clip(arr[:,:,0] * 1.30 - 20, 0, 255)
    g = np.clip(arr[:,:,1] * 0.85 + 10, 0, 255)
    b = np.clip(arr[:,:,2] * 1.15 + 20, 0, 255)
    img = Image.fromarray(np.stack([r, g, b], axis=2).astype(np.uint8))
    img = ImageEnhance.Contrast(img).enhance(1.35)
    img = ImageEnhance.Color(img).enhance(1.50)
    return img
def apply_cyberpunk(img):
    img = img.convert("RGB"); arr = np.array(img, dtype=np.float32)
    arr[:,:,0] = np.clip(arr[:,:,0]*0.85+20, 0, 255)
    arr[:,:,1] = np.clip(arr[:,:,1]*0.75, 0, 255)
    arr[:,:,2] = np.clip(arr[:,:,2]*1.30+15, 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))
    return ImageEnhance.Contrast(img).enhance(1.50)
def apply_golden_hour(img):
    img = img.convert("RGB"); arr = np.array(img, dtype=np.float32)
    arr[:,:,0] = np.clip(arr[:,:,0]*1.18+10, 0, 255)
    arr[:,:,1] = np.clip(arr[:,:,1]*1.08+5, 0, 255)
    arr[:,:,2] = np.clip(arr[:,:,2]*0.78, 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))
    return ImageEnhance.Brightness(img).enhance(1.12)
def apply_moonlight(img):
    img = img.convert("RGB"); arr = np.array(img, dtype=np.float32)
    arr[:,:,0] = np.clip(arr[:,:,0]*0.72, 0, 255)
    arr[:,:,1] = np.clip(arr[:,:,1]*0.82, 0, 255)
    arr[:,:,2] = np.clip(arr[:,:,2]*1.25+20, 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))
    img = ImageEnhance.Brightness(img).enhance(0.78)
    return ImageEnhance.Color(img).enhance(0.70)
def apply_polaroid(img):
    img = img.convert("RGB"); arr = np.array(img, dtype=np.float32)
    arr[:,:,0] = np.clip(arr[:,:,0]*1.05+8, 0, 255)
    arr[:,:,1] = np.clip(arr[:,:,1]*1.02+5, 0, 255)
    arr[:,:,2] = np.clip(arr[:,:,2]*0.90, 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))
    img = ImageEnhance.Contrast(img).enhance(0.88)
    return ImageEnhance.Brightness(img).enhance(1.08)
def apply_lomo(img):
    img = img.convert("RGB")
    img = ImageEnhance.Color(img).enhance(1.80)
    img = ImageEnhance.Contrast(img).enhance(1.40)
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    Y, X = np.ogrid[:h, :w]
    cx, cy = w/2, h/2
    r = np.sqrt((X-cx)**2 + (Y-cy)**2) / np.sqrt(cx**2+cy**2)
    vignette = np.clip(1.0 - r*0.6, 0.3, 1.0)
    for c in range(3): arr[:,:,c] *= vignette
    return Image.fromarray(np.clip(arr,0,255).astype(np.uint8))
def apply_velvet(img):
    img = img.convert("RGB")
    img = ImageEnhance.Brightness(img).enhance(0.80)
    img = ImageEnhance.Contrast(img).enhance(1.20)
    arr = np.array(img, dtype=np.float32)
    arr = arr*0.90 + 15
    img = Image.fromarray(np.clip(arr,0,255).astype(np.uint8))
    return ImageEnhance.Color(img).enhance(0.80)
def apply_arctic(img):
    img = img.convert("RGB"); arr = np.array(img, dtype=np.float32)
    arr[:,:,0] = np.clip(arr[:,:,0]*0.85+30, 0, 255)
    arr[:,:,1] = np.clip(arr[:,:,1]*0.92+25, 0, 255)
    arr[:,:,2] = np.clip(arr[:,:,2]*1.05+35, 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))
    img = ImageEnhance.Brightness(img).enhance(1.15)
    return ImageEnhance.Color(img).enhance(0.60)
def apply_cherry_blossom(img):
    img = img.convert("RGB"); arr = np.array(img, dtype=np.float32)
    arr[:,:,0] = np.clip(arr[:,:,0]*1.15+15, 0, 255)
    arr[:,:,1] = np.clip(arr[:,:,1]*0.88, 0, 255)
    arr[:,:,2] = np.clip(arr[:,:,2]*1.08+10, 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))
    img = ImageEnhance.Brightness(img).enhance(1.10)
    return ImageEnhance.Color(img).enhance(0.85)
def apply_emerald(img):
    img = img.convert("RGB"); arr = np.array(img, dtype=np.float32)
    arr[:,:,0] = np.clip(arr[:,:,0]*0.75, 0, 255)
    arr[:,:,1] = np.clip(arr[:,:,1]*1.25+10, 0, 255)
    arr[:,:,2] = np.clip(arr[:,:,2]*0.85, 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))
    return ImageEnhance.Color(img).enhance(1.30)
def apply_lavender(img):
    img = img.convert("RGB"); arr = np.array(img, dtype=np.float32)
    arr[:,:,0] = np.clip(arr[:,:,0]*1.05+15, 0, 255)
    arr[:,:,1] = np.clip(arr[:,:,1]*0.88+10, 0, 255)
    arr[:,:,2] = np.clip(arr[:,:,2]*1.15+20, 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))
    img = ImageEnhance.Brightness(img).enhance(1.10)
    return ImageEnhance.Color(img).enhance(0.80)
def apply_burnt_orange(img):
    img = img.convert("RGB"); arr = np.array(img, dtype=np.float32)
    arr[:,:,0] = np.clip(arr[:,:,0]*1.25+15, 0, 255)
    arr[:,:,1] = np.clip(arr[:,:,1]*0.92, 0, 255)
    arr[:,:,2] = np.clip(arr[:,:,2]*0.65, 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))
    return ImageEnhance.Contrast(img).enhance(1.15)
def apply_film_noir(img):
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(1.80)
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, 8, arr.shape)
    arr = np.clip(arr + noise, 0, 255)
    return Image.fromarray(arr.astype(np.uint8)).convert("RGB")
def apply_retro_70s(img):
    img = img.convert("RGB"); arr = np.array(img, dtype=np.float32)
    arr[:,:,0] = np.clip(arr[:,:,0]*1.12+10, 0, 255)
    arr[:,:,1] = np.clip(arr[:,:,1]*1.05+8, 0, 255)
    arr[:,:,2] = np.clip(arr[:,:,2]*0.70, 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))
    img = ImageEnhance.Contrast(img).enhance(0.85)
    return ImageEnhance.Color(img).enhance(1.15)
def apply_vaporwave(img):
    img = img.convert("RGB"); arr = np.array(img, dtype=np.float32)
    arr[:,:,0] = np.clip(arr[:,:,0]*1.20+25, 0, 255)
    arr[:,:,1] = np.clip(arr[:,:,1]*0.70, 0, 255)
    arr[:,:,2] = np.clip(arr[:,:,2]*1.30+20, 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))
    img = ImageEnhance.Color(img).enhance(1.60)
    return ImageEnhance.Contrast(img).enhance(1.20)
def apply_desert_sand(img):
    img = img.convert("RGB"); arr = np.array(img, dtype=np.float32)
    arr[:,:,0] = np.clip(arr[:,:,0]*1.10+12, 0, 255)
    arr[:,:,1] = np.clip(arr[:,:,1]*1.02+8, 0, 255)
    arr[:,:,2] = np.clip(arr[:,:,2]*0.82, 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))
    img = ImageEnhance.Color(img).enhance(0.70)
    return ImageEnhance.Brightness(img).enhance(1.08)
def apply_ocean_breeze(img):
    img = img.convert("RGB"); arr = np.array(img, dtype=np.float32)
    arr[:,:,0] = np.clip(arr[:,:,0]*0.82, 0, 255)
    arr[:,:,1] = np.clip(arr[:,:,1]*1.08+8, 0, 255)
    arr[:,:,2] = np.clip(arr[:,:,2]*1.20+12, 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))
    return ImageEnhance.Brightness(img).enhance(1.06)
def apply_autumn(img):
    img = img.convert("RGB"); arr = np.array(img, dtype=np.float32)
    arr[:,:,0] = np.clip(arr[:,:,0]*1.22+10, 0, 255)
    arr[:,:,1] = np.clip(arr[:,:,1]*0.95, 0, 255)
    arr[:,:,2] = np.clip(arr[:,:,2]*0.72, 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))
    img = ImageEnhance.Color(img).enhance(1.25)
    return ImageEnhance.Contrast(img).enhance(1.10)
def apply_infrared(img):
    img = img.convert("RGB"); arr = np.array(img, dtype=np.float32)
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    new_r = np.clip(b*0.8 + g*0.5, 0, 255)
    new_g = np.clip(r*0.2 + g*0.2 + b*0.8, 0, 255)
    new_b = np.clip(r*0.9, 0, 255)
    img = Image.fromarray(np.stack([new_r, new_g, new_b], axis=2).astype(np.uint8))
    return ImageEnhance.Contrast(img).enhance(1.20)
def apply_noir_gold(img):
    gray = np.array(img.convert("L"), dtype=np.float32)
    img = ImageEnhance.Contrast(Image.fromarray(gray.astype(np.uint8))).enhance(1.40)
    gray = np.array(img, dtype=np.float32)
    r = np.clip(gray*1.10+15, 0, 255)
    g = np.clip(gray*0.95+8, 0, 255)
    b = np.clip(gray*0.70, 0, 255)
    return Image.fromarray(np.stack([r,g,b], axis=2).astype(np.uint8))
def apply_pastel(img):
    img = img.convert("RGB"); arr = np.array(img, dtype=np.float32)
    arr = arr * 0.60 + 100
    img = Image.fromarray(np.clip(arr,0,255).astype(np.uint8))
    img = ImageEnhance.Color(img).enhance(0.70)
    return ImageEnhance.Brightness(img).enhance(1.10)
def apply_hdr(img):
    img = img.convert("RGB")
    img = ImageEnhance.Contrast(img).enhance(1.50)
    img = ImageEnhance.Sharpness(img).enhance(1.80)
    img = ImageEnhance.Color(img).enhance(1.40)
    return ImageEnhance.Brightness(img).enhance(1.05)
def apply_bleach_bypass(img):
    img = img.convert("RGB")
    gray = img.convert("L").convert("RGB")
    img = Image.blend(img, gray, alpha=0.45)
    img = ImageEnhance.Contrast(img).enhance(1.50)
    return ImageEnhance.Brightness(img).enhance(0.92)
def apply_rose_gold(img):
    img = img.convert("RGB"); arr = np.array(img, dtype=np.float32)
    arr[:,:,0] = np.clip(arr[:,:,0]*1.15+18, 0, 255)
    arr[:,:,1] = np.clip(arr[:,:,1]*0.92+8, 0, 255)
    arr[:,:,2] = np.clip(arr[:,:,2]*0.90+5, 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))
    return ImageEnhance.Brightness(img).enhance(1.08)
def apply_midnight(img):
    img = img.convert("RGB"); arr = np.array(img, dtype=np.float32)
    arr[:,:,0] = np.clip(arr[:,:,0]*0.55, 0, 255)
    arr[:,:,1] = np.clip(arr[:,:,1]*0.55, 0, 255)
    arr[:,:,2] = np.clip(arr[:,:,2]*0.90+20, 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))
    return ImageEnhance.Contrast(img).enhance(1.30)
def apply_candy_pop(img):
    img = img.convert("RGB")
    img = ImageEnhance.Color(img).enhance(2.50)
    img = ImageEnhance.Brightness(img).enhance(1.15)
    return ImageEnhance.Contrast(img).enhance(1.20)
def apply_sepia_classic(img):
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    r = arr[:,:,0]*0.393 + arr[:,:,1]*0.769 + arr[:,:,2]*0.189
    g = arr[:,:,0]*0.349 + arr[:,:,1]*0.686 + arr[:,:,2]*0.168
    b = arr[:,:,0]*0.272 + arr[:,:,1]*0.534 + arr[:,:,2]*0.131
    return Image.fromarray(np.clip(np.stack([r,g,b],axis=2),0,255).astype(np.uint8))
def apply_chrome(img):
    img = img.convert("RGB")
    img = ImageEnhance.Color(img).enhance(0.30)
    img = ImageEnhance.Contrast(img).enhance(1.45)
    img = ImageEnhance.Brightness(img).enhance(1.15)
    return ImageEnhance.Sharpness(img).enhance(1.50)
def apply_foggy_morning(img):
    img = img.convert("RGB")
    blurred = img.filter(ImageFilter.GaussianBlur(radius=2.5))
    img = Image.blend(img, blurred, alpha=0.40)
    img = ImageEnhance.Brightness(img).enhance(1.20)
    img = ImageEnhance.Contrast(img).enhance(0.75)
    return ImageEnhance.Color(img).enhance(0.80)
def apply_tropical(img):
    img = img.convert("RGB"); arr = np.array(img, dtype=np.float32)
    arr[:,:,0] = np.clip(arr[:,:,0]*1.10, 0, 255)
    arr[:,:,1] = np.clip(arr[:,:,1]*1.18+10, 0, 255)
    arr[:,:,2] = np.clip(arr[:,:,2]*0.92, 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))
    img = ImageEnhance.Color(img).enhance(1.60)
    return ImageEnhance.Brightness(img).enhance(1.08)
def apply_electric_blue(img):
    img = img.convert("RGB"); arr = np.array(img, dtype=np.float32)
    arr[:,:,0] = np.clip(arr[:,:,0]*0.65, 0, 255)
    arr[:,:,1] = np.clip(arr[:,:,1]*0.80, 0, 255)
    arr[:,:,2] = np.clip(arr[:,:,2]*1.40+25, 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))
    return ImageEnhance.Contrast(img).enhance(1.30)
def apply_copper(img):
    img = img.convert("RGB"); arr = np.array(img, dtype=np.float32)
    arr[:,:,0] = np.clip(arr[:,:,0]*1.20+15, 0, 255)
    arr[:,:,1] = np.clip(arr[:,:,1]*0.88+5, 0, 255)
    arr[:,:,2] = np.clip(arr[:,:,2]*0.68, 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))
    img = ImageEnhance.Contrast(img).enhance(1.10)
    return ImageEnhance.Color(img).enhance(0.90)
def apply_dream_glow(img):
    img = img.convert("RGB")
    blurred = img.filter(ImageFilter.GaussianBlur(radius=4))
    bright = ImageEnhance.Brightness(blurred).enhance(1.30)
    img = Image.blend(img, bright, alpha=0.35)
    return ImageEnhance.Color(img).enhance(0.90)
def apply_duotone_purple(img):
    gray = np.array(img.convert("L"), dtype=np.float32) / 255.0
    r = np.clip(gray * 180 + 40, 0, 255)
    g = np.clip(gray * 60, 0, 255)
    b = np.clip(gray * 220 + 30, 0, 255)
    return Image.fromarray(np.stack([r,g,b], axis=2).astype(np.uint8))
def apply_duotone_teal(img):
    gray = np.array(img.convert("L"), dtype=np.float32) / 255.0
    r = np.clip(gray * 40, 0, 255)
    g = np.clip(gray * 200 + 30, 0, 255)
    b = np.clip(gray * 190 + 40, 0, 255)
    return Image.fromarray(np.stack([r,g,b], axis=2).astype(np.uint8))
def apply_noir_crimson(img):
    gray = np.array(img.convert("L"), dtype=np.float32)
    gray = np.clip(gray * 1.30, 0, 255)
    r = np.clip(gray * 1.15 + 15, 0, 255)
    g = np.clip(gray * 0.65, 0, 255)
    b = np.clip(gray * 0.60, 0, 255)
    return Image.fromarray(np.stack([r,g,b], axis=2).astype(np.uint8))
FILTER_FN = {
    "Vintage":        apply_vintage,
    "B&W":            apply_bw,
    "Warm Tone":      apply_warm,
    "Cool Tone":      apply_cool,
    "Bright Boost":   apply_bright_boost,
    "Sunset Glow":    apply_sunset_glow,
    "Aesthetic Pink": apply_aesthetic_pink,
    "Matte":          apply_matte,
    "Neon Glow":      apply_neon_glow,
    "Cinematic":      apply_cinematic,
    "Soft Focus":     apply_soft_focus,
    "Faded":          apply_faded,
    "Dramatic":       apply_dramatic,
    "Aqua":           apply_aqua,
    "Cross Process":  apply_cross_process,
    "Cyberpunk":      apply_cyberpunk,
    "Golden Hour":    apply_golden_hour,
    "Moonlight":      apply_moonlight,
    "Polaroid":       apply_polaroid,
    "Lomo":           apply_lomo,
    "Velvet":         apply_velvet,
    "Arctic":         apply_arctic,
    "Cherry Blossom": apply_cherry_blossom,
    "Emerald":        apply_emerald,
    "Lavender":       apply_lavender,
    "Burnt Orange":   apply_burnt_orange,
    "Film Noir":      apply_film_noir,
    "Retro 70s":      apply_retro_70s,
    "Vaporwave":      apply_vaporwave,
    "Desert Sand":    apply_desert_sand,
    "Ocean Breeze":   apply_ocean_breeze,
    "Autumn":         apply_autumn,
    "Infrared":       apply_infrared,
    "Noir Gold":      apply_noir_gold,
    "Pastel":         apply_pastel,
    "HDR Effect":     apply_hdr,
    "Bleach Bypass":  apply_bleach_bypass,
    "Rose Gold":      apply_rose_gold,
    "Midnight":       apply_midnight,
    "Candy Pop":      apply_candy_pop,
    "Sepia Classic":  apply_sepia_classic,
    "Chrome":         apply_chrome,
    "Foggy Morning":  apply_foggy_morning,
    "Tropical":       apply_tropical,
    "Electric Blue":  apply_electric_blue,
    "Copper":         apply_copper,
    "Dream Glow":     apply_dream_glow,
    "Duotone Purple": apply_duotone_purple,
    "Duotone Teal":   apply_duotone_teal,
    "Noir Crimson":   apply_noir_crimson,
}
def apply_filter(pil_image: Image.Image, filter_name: str) -> Image.Image:
    fn = FILTER_FN.get(filter_name)
    if fn:
        return fn(pil_image)
    return pil_image
                  
def pil_to_ctk(pil_image: Image.Image, size: tuple) -> ctk.CTkImage:
    resized = pil_image.copy()
    resized.thumbnail(size, Image.LANCZOS)
    return ctk.CTkImage(light_image=resized, dark_image=resized, size=resized.size)
def make_thumbnail(pil_image: Image.Image, filter_name: str, size=(160, 120)) -> ctk.CTkImage:
    filtered = apply_filter(pil_image, filter_name)
    filtered.thumbnail(size, Image.LANCZOS)
    return ctk.CTkImage(light_image=filtered, dark_image=filtered, size=filtered.size)
                         
class FilterApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(" Instagram Filter Recommender  |  AI-Powered")
        self.geometry("1280x820")
        self.minsize(1100, 720)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        self.configure(fg_color=DARK_BG)
        self.original_image: Image.Image | None = None
        self.current_filtered: Image.Image | None = None
        self.features: dict = {}
        self.ranked_filters: list = []
        self.selected_filter: str | None = None
        self.filter_card_frames: dict = {}
        self.filter_thumbnails: dict = {}                          
        self._build_layout()
                                                                
    def _build_layout(self):
                        
        self._build_header()
        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.pack(fill="both", expand=True, padx=20, pady=(0, 16))
        self.content.columnconfigure(0, weight=3)
        self.content.columnconfigure(1, weight=4)
        self.content.rowconfigure(0, weight=1)
        self._build_left_panel()
        self._build_right_panel()
    def _build_header(self):
        hdr = ctk.CTkFrame(self, fg_color=CARD_BG, corner_radius=0, height=68)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        left = ctk.CTkFrame(hdr, fg_color="transparent")
        left.pack(side="left", padx=24, pady=10)
        ctk.CTkLabel(
            left, text="  Filter Recommender",
            font=ctk.CTkFont("Courier New", 22, "bold"),
            text_color=ACCENT_PURPLE
        ).pack(side="left")
        ctk.CTkLabel(
            left, text="    AI-Powered Instagram Filter Matching",
            font=ctk.CTkFont("Courier New", 12),
            text_color=TEXT_MUTED
        ).pack(side="left", padx=(6, 0))
        self.theme_var = ctk.StringVar(value="Dark")
        ctk.CTkSegmentedButton(
            hdr,
            values=["Dark", "Light"],
            variable=self.theme_var,
            command=self._toggle_theme,
            font=ctk.CTkFont("Courier New", 12),
            fg_color=DARK_BG,
            selected_color=ACCENT_PURPLE,
            selected_hover_color="#7C3AED",
            unselected_color=DARK_BG,
            unselected_hover_color=BORDER,
        ).pack(side="right", padx=24, pady=18)
    def _build_left_panel(self):
        self.left = ctk.CTkFrame(self.content, fg_color=CARD_BG, corner_radius=16)
        self.left.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=8)
        self.left.rowconfigure(1, weight=1)
        self.left.columnconfigure(0, weight=1)
        ctk.CTkLabel(
            self.left, text="  Image Upload & Preview",
            font=ctk.CTkFont("Courier New", 14, "bold"),
            text_color=TEXT_PRIMARY
        ).grid(row=0, column=0, sticky="w", padx=18, pady=(16, 8))
        self.preview_frame = ctk.CTkFrame(
            self.left, fg_color=DARK_BG, corner_radius=12,
            border_width=2, border_color=BORDER
        )
        self.preview_frame.grid(row=1, column=0, sticky="nsew", padx=14, pady=4)
        self.preview_frame.rowconfigure(0, weight=1)
        self.preview_frame.columnconfigure(0, weight=1)
        self.upload_prompt = ctk.CTkFrame(self.preview_frame, fg_color="transparent")
        self.upload_prompt.grid(row=0, column=0)
        ctk.CTkLabel(self.upload_prompt, text="",
                      font=ctk.CTkFont(size=52)).pack(pady=(30, 8))
        ctk.CTkLabel(self.upload_prompt, text="Drop or click to upload",
                      font=ctk.CTkFont("Courier New", 14),
                      text_color=TEXT_MUTED).pack()
        ctk.CTkLabel(self.upload_prompt, text="JPG  PNG  WEBP  BMP",
                      font=ctk.CTkFont("Courier New", 10),
                      text_color=TEXT_MUTED).pack(pady=(4, 0))
        self.img_label = ctk.CTkLabel(self.preview_frame, text="")
        btn_row = ctk.CTkFrame(self.left, fg_color="transparent")
        btn_row.grid(row=2, column=0, pady=(8, 6), padx=14, sticky="ew")
        btn_row.columnconfigure((0, 1, 2), weight=1)
        ctk.CTkButton(
            btn_row, text="  Upload Image",
            command=self._upload_image,
            font=ctk.CTkFont("Courier New", 13, "bold"),
            fg_color=ACCENT_PURPLE, hover_color="#7C3AED",
            corner_radius=10, height=40
        ).grid(row=0, column=0, padx=(0, 5), sticky="ew")
        self.analyze_btn = ctk.CTkButton(
            btn_row, text="  Analyze",
            command=self._start_analysis,
            font=ctk.CTkFont("Courier New", 13, "bold"),
            fg_color=ACCENT_CYAN, hover_color="#0891B2",
            corner_radius=10, height=40, state="disabled"
        )
        self.analyze_btn.grid(row=0, column=1, padx=5, sticky="ew")
        self.save_btn = ctk.CTkButton(
            btn_row, text="  Save",
            command=self._save_image,
            font=ctk.CTkFont("Courier New", 13, "bold"),
            fg_color=SUCCESS, hover_color="#059669",
            corner_radius=10, height=40, state="disabled"
        )
        self.save_btn.grid(row=0, column=2, padx=(5, 0), sticky="ew")
        self.progress = ctk.CTkProgressBar(
            self.left, mode="indeterminate",
            progress_color=ACCENT_PURPLE, fg_color=BORDER, height=4
        )
        self.progress.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 4))
        self.progress.set(0)
        ctk.CTkLabel(
            self.left, text="  Image Metrics",
            font=ctk.CTkFont("Courier New", 13, "bold"),
            text_color=TEXT_PRIMARY
        ).grid(row=4, column=0, sticky="w", padx=18, pady=(8, 4))
        self.details_frame = ctk.CTkFrame(self.left, fg_color=DARK_BG, corner_radius=10)
        self.details_frame.grid(row=5, column=0, sticky="ew", padx=14, pady=(0, 14))
        self.detail_labels = {}
        metrics = [
            ("brightness", "  Brightness"),
            ("contrast",   "  Contrast"),
            ("saturation", "  Saturation"),
            ("dom_color",  "  Dominant Color"),
            ("mood",       "  Mood"),
        ]
        for i, (key, label) in enumerate(metrics):
            row_f = ctk.CTkFrame(self.details_frame, fg_color="transparent")
            row_f.pack(fill="x", padx=10, pady=3)
            ctk.CTkLabel(row_f, text=label,
                          font=ctk.CTkFont("Courier New", 11),
                          text_color=TEXT_MUTED, width=140, anchor="w").pack(side="left")
            val = ctk.CTkLabel(row_f, text="",
                                font=ctk.CTkFont("Courier New", 11, "bold"),
                                text_color=TEXT_PRIMARY, anchor="w")
            val.pack(side="left")
            self.detail_labels[key] = val
    def _build_right_panel(self):
        self.right = ctk.CTkFrame(self.content, fg_color=CARD_BG, corner_radius=16)
        self.right.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=8)
        self.right.columnconfigure(0, weight=1)
        self.right.rowconfigure(1, weight=1)
        self.right.rowconfigure(3, weight=2)
        ctk.CTkLabel(
            self.right, text="  AI Filter Recommendations",
            font=ctk.CTkFont("Courier New", 14, "bold"),
            text_color=TEXT_PRIMARY
        ).grid(row=0, column=0, sticky="w", padx=18, pady=(16, 8))
        self.cards_scroll = ctk.CTkScrollableFrame(
            self.right, fg_color="transparent", height=210
        )
        self.cards_scroll.grid(row=1, column=0, sticky="nsew", padx=14, pady=4)
        self.cards_scroll.columnconfigure((0, 1, 2), weight=1)
        self._render_placeholder_cards()
        ba_hdr = ctk.CTkFrame(self.right, fg_color="transparent")
        ba_hdr.grid(row=2, column=0, sticky="ew", padx=18, pady=(10, 4))
        ctk.CTkLabel(
            ba_hdr, text="  Before vs After",
            font=ctk.CTkFont("Courier New", 14, "bold"),
            text_color=TEXT_PRIMARY
        ).pack(side="left")
        self.compare_btn = ctk.CTkButton(
            ba_hdr, text="  Compare All",
            command=self._open_compare_window,
            font=ctk.CTkFont("Courier New", 11),
            fg_color=DARK_BG, hover_color=BORDER,
            border_color=BORDER, border_width=1,
            corner_radius=8, height=28, width=110, state="disabled"
        )
        self.compare_btn.pack(side="right")
        ba_frame = ctk.CTkFrame(self.right, fg_color=DARK_BG, corner_radius=12)
        ba_frame.grid(row=3, column=0, sticky="nsew", padx=14, pady=(0, 14))
        ba_frame.columnconfigure((0, 1), weight=1)
        ba_frame.rowconfigure(1, weight=1)
        ctk.CTkLabel(ba_frame, text="Original",
                      font=ctk.CTkFont("Courier New", 11),
                      text_color=TEXT_MUTED).grid(row=0, column=0, pady=(8, 4))
        ctk.CTkLabel(ba_frame, text="Filtered",
                      font=ctk.CTkFont("Courier New", 11),
                      text_color=TEXT_MUTED).grid(row=0, column=1, pady=(8, 4))
        orig_box = ctk.CTkFrame(ba_frame, fg_color=CARD_BG, corner_radius=8)
        orig_box.grid(row=1, column=0, padx=(10, 5), pady=(0, 12), sticky="nsew")
        self.ba_orig_label = ctk.CTkLabel(orig_box, text="", fg_color="transparent")
        self.ba_orig_label.pack(expand=True, fill="both", padx=4, pady=4)
        filt_box = ctk.CTkFrame(ba_frame, fg_color=CARD_BG, corner_radius=8)
        filt_box.grid(row=1, column=1, padx=(5, 10), pady=(0, 12), sticky="nsew")
        self.ba_filt_label = ctk.CTkLabel(filt_box, text="", fg_color="transparent")
        self.ba_filt_label.pack(expand=True, fill="both", padx=4, pady=4)
        for lbl in (self.ba_orig_label, self.ba_filt_label):
            lbl.configure(text=" upload an image ",
                           font=ctk.CTkFont("Courier New", 11),
                           text_color=TEXT_MUTED)
                                                                
    def _render_placeholder_cards(self):
        for widget in self.cards_scroll.winfo_children():
            widget.destroy()
        for i in range(3):
            card = ctk.CTkFrame(
                self.cards_scroll, fg_color=DARK_BG,
                corner_radius=12, border_width=1, border_color=BORDER
            )
            card.grid(row=0, column=i, padx=6, pady=4, sticky="nsew")
            ctk.CTkLabel(card, text=f"Filter {i+1}",
                          font=ctk.CTkFont("Courier New", 12),
                          text_color=TEXT_MUTED).pack(pady=20)
            ctk.CTkLabel(card, text="Upload & Analyze ",
                          font=ctk.CTkFont("Courier New", 10),
                          text_color=BORDER).pack()
            ctk.CTkProgressBar(card, progress_color=BORDER, fg_color=BORDER,
                                height=3).pack(fill="x", padx=10, pady=12)
                                                                
    def _upload_image(self):
        path = filedialog.askopenfilename(
            title="Choose an Image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff")]
        )
        if not path:
            return
        self.original_image = Image.open(path).convert("RGB")
        self._show_preview(self.original_image)
        self.analyze_btn.configure(state="normal")
        self.selected_filter = None
        self.current_filtered = None
        self._render_placeholder_cards()
        self._reset_ba_previews()
    def _show_preview(self, img: Image.Image):
        self.upload_prompt.grid_forget()
        thumb = pil_to_ctk(img, (400, 320))
        self.img_label.configure(image=thumb, text="")
        self.img_label.image = thumb
        self.img_label.grid(row=0, column=0, padx=8, pady=8)
    def _start_analysis(self):
        if self.original_image is None:
            return
        self.analyze_btn.configure(state="disabled", text="  Analyzing")
        self.progress.configure(mode="indeterminate")
        self.progress.start()
        threading.Thread(target=self._run_analysis, daemon=True).start()
    def _run_analysis(self):
        try:
            self.features = extract_features(self.original_image)
            self.ranked_filters = recommend_filters(self.features)
            time.sleep(0.4)                                
            self.after(0, self._on_analysis_done)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.after(0, self._analysis_cleanup)
    def _on_analysis_done(self):
        self._analysis_cleanup()
        self._update_detail_labels()
        self._render_filter_cards()
        self.compare_btn.configure(state="normal")
    def _analysis_cleanup(self):
        self.progress.stop()
        self.progress.set(0)
        self.analyze_btn.configure(state="normal", text="  Analyze")
                                                                
    def _update_detail_labels(self):
        f = self.features
        b = f["brightness"]
        s = f["saturation"]
        wr = f["warm_ratio"]
        brightness_text = (
            "Very Dark" if b < 60 else
            "Dark"      if b < 100 else
            "Medium"    if b < 160 else
            "Bright"    if b < 210 else
            "Very Bright"
        )
        self.detail_labels["brightness"].configure(text=f"{brightness_text}  ({b:.0f})")
        self.detail_labels["contrast"].configure(
            text=f"{f['contrast']:.1f}  {'High' if f['contrast']>55 else 'Medium' if f['contrast']>35 else 'Low'}"
        )
        self.detail_labels["saturation"].configure(
            text=f"{s:.1f}  {'Vivid' if s>130 else 'Moderate' if s>70 else 'Muted'}"
        )
        hex_color = get_dominant_color_hex(f["dominant_colors"])
        self.detail_labels["dom_color"].configure(text=hex_color)
        mood = (
            " Warm & Vibrant" if wr > 0.6 and s > 100 else
            " Warm"           if wr > 0.5 else
            " Cool"           if wr < 0.3 else
            " Neutral"
        )
        self.detail_labels["mood"].configure(text=mood)
                                                                
    def _render_filter_cards(self):
        for w in self.cards_scroll.winfo_children():
            w.destroy()
        self.filter_card_frames.clear()
        self.filter_thumbnails.clear()
        top3 = self.ranked_filters[:3]
        accent_colors = [ACCENT_PURPLE, ACCENT_PINK, ACCENT_CYAN]
        medals = ["", "", ""]
        for col, (fname, score) in enumerate(top3):
            info = FILTERS[fname]
            accent = accent_colors[col]
            card = ctk.CTkFrame(
                self.cards_scroll, fg_color=DARK_BG,
                corner_radius=14, border_width=2, border_color=accent
            )
            card.grid(row=0, column=col, padx=6, pady=4, sticky="nsew")
            self.cards_scroll.columnconfigure(col, weight=1)
            self.filter_card_frames[fname] = card
            ctk.CTkLabel(card, text=f"{medals[col]} {info['emoji']}",
                          font=ctk.CTkFont(size=22)).pack(pady=(12, 2))
            ctk.CTkLabel(card, text=fname,
                          font=ctk.CTkFont("Courier New", 13, "bold"),
                          text_color=accent).pack()
            ctk.CTkLabel(card, text=info["desc"],
                          font=ctk.CTkFont("Courier New", 10),
                          text_color=TEXT_MUTED,
                          wraplength=140).pack(padx=8, pady=4)
            bar = ctk.CTkProgressBar(card, progress_color=accent,
                                      fg_color=BORDER, height=6, corner_radius=3)
            bar.pack(fill="x", padx=12, pady=(4, 2))
            bar.set(score / 100)
            ctk.CTkLabel(card, text=f"{score:.1f}% match",
                          font=ctk.CTkFont("Courier New", 11, "bold"),
                          text_color=accent).pack()
            thumb = make_thumbnail(self.original_image, fname, (150, 110))
            self.filter_thumbnails[fname] = thumb
            thumb_lbl = ctk.CTkLabel(card, image=thumb, text="")
            thumb_lbl.pack(padx=8, pady=8)
            ctk.CTkButton(
                card, text="Apply Filter",
                command=lambda fn=fname: self._apply_filter(fn),
                font=ctk.CTkFont("Courier New", 11, "bold"),
                fg_color=accent, hover_color="#555",
                corner_radius=8, height=32
            ).pack(padx=12, pady=(0, 12), fill="x")
                                                                
    def _apply_filter(self, filter_name: str):
        if self.original_image is None:
            return
        self.selected_filter = filter_name
        self.current_filtered = apply_filter(self.original_image, filter_name)
        orig_thumb = pil_to_ctk(self.original_image, (280, 200))
        filt_thumb = pil_to_ctk(self.current_filtered, (280, 200))
        self.ba_orig_label.configure(image=orig_thumb, text="")
        self.ba_orig_label.image = orig_thumb
        self.ba_filt_label.configure(image=filt_thumb, text="")
        self.ba_filt_label.image = filt_thumb
        self._show_preview(self.current_filtered)
        self.save_btn.configure(state="normal")
    def _open_fullscreen(self, event=None):
        img = self.current_filtered if self.current_filtered else self.original_image
        if img is None:
            return
        fs = ctk.CTkToplevel(self)
        fs.title("  Full Screen Preview    Press Escape or double-click to close")
        fs.attributes("-fullscreen", True)
        fs.configure(fg_color="#000000")
        fs.grab_set()
        fs.focus_force()
        sw = fs.winfo_screenwidth()
        sh = fs.winfo_screenheight()
        display = img.copy()
        display.thumbnail((sw, sh), Image.LANCZOS)
        ctk_img = ctk.CTkImage(light_image=display, dark_image=display, size=display.size)
        lbl = ctk.CTkLabel(fs, image=ctk_img, text="", fg_color="#000000")
        lbl.image = ctk_img
        lbl.pack(expand=True, fill="both")
        filter_text = f"Filter: {self.selected_filter}" if self.selected_filter else "Original"
        overlay = ctk.CTkLabel(
            fs, text=f"  {filter_text}    Double-click or Esc to close  ",
            font=ctk.CTkFont("Courier New", 12),
            text_color=TEXT_MUTED, fg_color="#000000"
        )
        overlay.place(relx=0.5, rely=0.97, anchor="center")
        def _close(e=None):
            fs.destroy()
        fs.bind("<Escape>", _close)
        fs.bind("<Double-Button-1>", _close)
        lbl.bind("<Double-Button-1>", _close)
    def _reset_ba_previews(self):
        for lbl in (self.ba_orig_label, self.ba_filt_label):
            lbl.configure(image=None, text=" upload an image ",
                           font=ctk.CTkFont("Courier New", 11),
                           text_color=TEXT_MUTED)
        self.save_btn.configure(state="disabled")
        self.compare_btn.configure(state="disabled")
                                                                
    def _save_image(self):
        if self.current_filtered is None:
            messagebox.showwarning("No Filter Applied", "Please apply a filter first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")],
            title="Save Filtered Image"
        )
        if path:
            self.current_filtered.save(path, quality=95)
            messagebox.showinfo("Saved ", f"Image saved to:\n{path}")
                                                                
    def _open_compare_window(self):
        if self.original_image is None:
            return
        win = ctk.CTkToplevel(self)
        win.title("  Compare All Filters    Click a filter to apply it")
        win.geometry("1200x700")
        win.configure(fg_color=DARK_BG)
        win.grab_set()
        hdr = ctk.CTkFrame(win, fg_color="transparent")
        hdr.pack(fill="x", padx=20, pady=(16, 4))
        ctk.CTkLabel(
            hdr, text="  Filter Comparison Gallery",
            font=ctk.CTkFont("Courier New", 18, "bold"),
            text_color=ACCENT_PURPLE
        ).pack(side="left")
        ctk.CTkLabel(
            hdr, text="    Click any filter to apply & close",
            font=ctk.CTkFont("Courier New", 11),
            text_color=TEXT_MUTED
        ).pack(side="left", padx=(8, 0))
        grid = ctk.CTkScrollableFrame(win, fg_color="transparent")
        grid.pack(fill="both", expand=True, padx=20, pady=(4, 16))
        all_filters = ["Original"] + list(FILTERS.keys())
        cols = 4
        max_score = max((s for _, s in self.ranked_filters), default=0)
        def _on_card_click(fname, window):
            window.grab_release()
            window.destroy()
            if fname != "Original":
                self.after(50, lambda: self._apply_filter(fname))
        for idx, fname in enumerate(all_filters):
            row, col = divmod(idx, cols)
            if fname == "Original":
                img = self.original_image.copy()
                label_text = "  Original"
                border = TEXT_MUTED
                score_text = ""
            else:
                img = apply_filter(self.original_image, fname)
                info = FILTERS[fname]
                score = next((s for n, s in self.ranked_filters if n == fname), 0)
                label_text = f"{info['emoji']}  {fname}"
                score_text = f"{score:.0f}% match"
                border = ACCENT_PURPLE if score == max_score else BORDER
            thumb = img.copy()
            thumb.thumbnail((210, 165), Image.LANCZOS)
            ctk_thumb = ctk.CTkImage(light_image=thumb, dark_image=thumb, size=thumb.size)
            card = ctk.CTkFrame(grid, fg_color=CARD_BG, corner_radius=12,
                                 border_width=2, border_color=border,
                                 cursor="hand2")
            card.grid(row=row, column=col, padx=8, pady=8, sticky="nsew")
            grid.columnconfigure(col, weight=1)
            img_lbl = ctk.CTkLabel(card, image=ctk_thumb, text="")
            img_lbl.image = ctk_thumb
            img_lbl.pack(padx=8, pady=(10, 4))
            ctk.CTkLabel(card, text=label_text,
                          font=ctk.CTkFont("Courier New", 11, "bold"),
                          text_color=TEXT_PRIMARY).pack(pady=(0, 2))
            if score_text:
                ctk.CTkLabel(card, text=score_text,
                              font=ctk.CTkFont("Courier New", 10),
                              text_color=ACCENT_CYAN if border == ACCENT_PURPLE else TEXT_MUTED
                              ).pack(pady=(0, 4))
            def _on_enter(e, c=card, b=border):
                c.configure(border_color=ACCENT_CYAN)
            def _on_leave(e, c=card, b=border):
                c.configure(border_color=b)
            for widget in [card, img_lbl]:
                widget.bind("<Button-1>", lambda e, fn=fname, w=win: _on_card_click(fn, w))
                widget.bind("<Enter>", _on_enter)
                widget.bind("<Leave>", _on_leave)
            btn_text = "View Original" if fname == "Original" else "Apply Filter"
            ctk.CTkButton(
                card, text=btn_text,
                command=lambda fn=fname, w=win: _on_card_click(fn, w),
                font=ctk.CTkFont("Courier New", 10, "bold"),
                fg_color=ACCENT_PURPLE if border == ACCENT_PURPLE else DARK_BG,
                hover_color="#7C3AED",
                border_color=border, border_width=1,
                corner_radius=6, height=26
            ).pack(padx=10, pady=(2, 10), fill="x")
                                                                
    def _toggle_theme(self, value: str):
        ctk.set_appearance_mode("dark" if value == "Dark" else "light")
              
if __name__ == "__main__":
    app = FilterApp()
    app.mainloop()
