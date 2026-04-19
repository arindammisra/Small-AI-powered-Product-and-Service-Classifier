# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 15:26:22 2026

@author: arind
UNSPSC Classifier
Image Enhanced

"""

import pandas as pd
import requests
import base64
import os
import re
import tkinter as tk
from tkinter import filedialog, scrolledtext
from rank_bm25 import BM25Okapi
from PIL import Image, ImageTk


EXCEL_PATH = r'C:\Users\arind\Downloads\Classifier UNSPSCv2\unspscfull.xlsx'
OLLAMA_API_URL = "http://localhost:11434"
LLM_MODEL = "gemma3:12b"
#LLM_MODEL = "gemma3:4b"
#LLM_MODEL = "qwen2.5vl:7b"


USE_IMAGE_MATCHING = True
BM25_THRESHOLD = 8.0


# --------------------------------------------------
# OLLAMA API
# --------------------------------------------------

def generate_text(prompt, model=LLM_MODEL, image_path=None):

    base_url = OLLAMA_API_URL.rstrip("/")

    if not base_url.endswith("/api"):
        base_url += "/api"

    url = f"{base_url}/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    if image_path and os.path.exists(image_path):

        try:
    
            # Open and compress image slightly to avoid Ollama crashes
            img = Image.open(image_path)
    
            img = img.convert("RGB")
    
            img.thumbnail((1024, 1024))
    
            from io import BytesIO
            buffer = BytesIO()
    
            img.save(buffer, format="JPEG", quality=85)
    
            image_bytes = buffer.getvalue()
    
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    
            payload["images"] = [image_base64]
    
        except Exception as e:
    
            return f"ERROR: Image processing failed: {e}"

    try:

        response = requests.post(url, json=payload, timeout=180)
        response.raise_for_status()

        return response.json()["response"].strip()

    except Exception as e:

        return f"ERROR: {e}"


# --------------------------------------------------
# IMAGE DESCRIPTION
# --------------------------------------------------

def describe_image(image_path):

    prompt = """
Describe the product in this image in 15 to 20 words
Use neutral commercial language suitable for UNSPSC classification.
"""

    return generate_text(prompt, image_path=image_path)


# --------------------------------------------------
# IMAGE MATCH VALIDATION
# --------------------------------------------------

def validate_image_description(user_desc, image_desc):

    prompt = f"""
You are validating whether an image matches a description.

User description:
{user_desc}

Image description:
{image_desc}

Do they refer to the same product?

Answer ONLY:
YES
or
NO
"""

    resp = generate_text(prompt)

    return "YES" in resp.upper()


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def clean_code(val):

    s = str(val).replace(".0", "").strip()
    return s.zfill(8) if s else ""


def preprocess(text):

    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    return text.split()


# --------------------------------------------------
# TREE BUILDER
# --------------------------------------------------

def build_trees(file_path):

    df = pd.read_excel(file_path)

    text_cols = df.select_dtypes(include=["object", "string"]).columns
    df[text_cols] = df[text_cols].fillna("")

    goods_tree = {}
    services_tree = {}

    segment_is_service = {}

    for _, row in df.iterrows():

        seg = clean_code(row.get("Segment"))

        title = str(row.get("Segment Title")).lower()
        definition = str(row.get("Segment Definition")).lower()

        segment_is_service[seg] = "service" in title or "service" in definition

    for _, row in df.iterrows():

        seg = clean_code(row.get("Segment"))
        fam = clean_code(row.get("Family"))
        cls = clean_code(row.get("Class"))
        com = clean_code(row.get("Commodity"))

        if not seg:
            continue

        target_tree = services_tree if segment_is_service.get(seg) else goods_tree

        seg_text = f"{row.get('Segment Title')} - {row.get('Segment Definition')}"
        fam_text = f"{row.get('Family Title')} - {row.get('Family Definition')}"
        cls_text = f"{row.get('Class Title')} - {row.get('Class Definition')}"
        com_text = f"{row.get('Commodity Title')} - {row.get('Commodity Definition')}"

        target_tree.setdefault(seg, {"text": seg_text, "children": {}})

        if fam:

            target_tree[seg]["children"].setdefault(
                fam, {"text": fam_text, "children": {}}
            )

            if cls:

                target_tree[seg]["children"][fam]["children"].setdefault(
                    cls, {"text": cls_text, "children": {}}
                )

                if com:

                    target_tree[seg]["children"][fam]["children"][cls]["children"][com] = {
                        "text": com_text
                    }

    return goods_tree, services_tree


# --------------------------------------------------
# BM25 BUILD
# --------------------------------------------------

def build_bm25(file_path):

    df = pd.read_excel(file_path)

    text_cols = df.select_dtypes(include=["object", "string"]).columns
    df[text_cols] = df[text_cols].fillna("")

    goods_meta = []
    services_meta = []

    for _, row in df.iterrows():

        seg_title = str(row.get("Segment Title")).lower()
        seg_def = str(row.get("Segment Definition")).lower()

        is_service = "service" in seg_title or "service" in seg_def

        code = clean_code(row.get("Commodity"))

        if not code:
            continue

        entry = {
            "code": code,
            "title": row.get("Commodity Title"),
            "definition": row.get("Commodity Definition")
        }

        if is_service:
            services_meta.append(entry)
        else:
            goods_meta.append(entry)

    goods_bm25 = BM25Okapi([preprocess(x["title"]) for x in goods_meta])
    services_bm25 = BM25Okapi([preprocess(x["title"]) for x in services_meta])

    return goods_bm25, goods_meta, services_bm25, services_meta


# --------------------------------------------------
# BM25 SEARCH
# --------------------------------------------------

def bm25_search(query, bm25, metadata):

    tokens = preprocess(query)

    scores = bm25.get_scores(tokens)

    idx = max(range(len(scores)), key=lambda i: scores[i])

    return metadata[idx], scores[idx]


# --------------------------------------------------
# EXPAND DESCRIPTION
# --------------------------------------------------

def expand_description(desc):

    prompt = f"""
Analyze the following product or service description:

"{desc}"

Rewrite it as a clear UNSPSC friendly commercial description and classify it as a good or service.

Rules:
- 15 to 20 words
- neutral commercial language
- infer missing context if description is short

Format:

Type: Good or Service
Expanded: rewritten description
"""

    resp = generate_text(prompt)

    item_type = "Service" if "service" in resp.lower() else "Good"

    expanded = resp.split("Expanded:")[-1].strip()

    return item_type, expanded


# --------------------------------------------------
# CHOOSE OPTION
# --------------------------------------------------

def choose_option(level, desc, options, log):

    if len(options) == 1:
        return list(options.keys())[0]

    options_text = "\n".join(
        f"{c} | {n['text']}"
        for c, n in options.items()
    )

    prompt = f"""
You are an expert UNSPSC classification specialist.

Item description:
{desc}

Choose the best matching {level} category.

Options:
{options_text}

Return ONLY the 8 digit code.
"""

    resp = generate_text(prompt)

    log(f"LLM raw response: {resp}")

    matches = re.findall(r"\b\d{8}\b", resp)

    for m in matches:
        if m in options:
            return m

    return list(options.keys())[0]


# --------------------------------------------------
# TREE TRAVERSAL
# --------------------------------------------------

def traverse_tree(desc, tree, log):

    log("\n[Level 1: Segment]")

    seg = choose_option("Segment", desc, tree, log)

    log(f"Selected Segment: {seg} - {tree[seg]['text']}")

    fams = tree[seg]["children"]

    log("\n[Level 2: Family]")

    fam = choose_option("Family", desc, fams, log)

    log(f"Selected Family: {fam} - {fams[fam]['text']}")

    classes = fams[fam]["children"]

    log("\n[Level 3: Class]")

    cls = choose_option("Class", desc, classes, log)

    log(f"Selected Class: {cls} - {classes[cls]['text']}")

    commodities = classes[cls]["children"]

    log("\n[Level 4: Commodity]")

    com = choose_option("Commodity", desc, commodities, log)

    log(f"Selected Commodity: {com} - {commodities[com]['text']}")

    return com


# --------------------------------------------------
# FIND DETAILS
# --------------------------------------------------

def find_details(tree, code):

    for seg in tree.values():

        for fam in seg["children"].values():

            for cls in fam["children"].values():

                if code in cls["children"]:

                    text = cls["children"][code]["text"]

                    parts = text.split(" - ", 1)

                    return {
                        "code": code,
                        "title": parts[0],
                        "definition": parts[1] if len(parts) > 1 else ""
                    }

    return None


# --------------------------------------------------
# FINAL DECISION
# --------------------------------------------------

def final_decision(desc, llm_choice, bm25_choice, bm25_score):

    prompt = f"""
You are a senior UNSPSC taxonomy expert.

Item:
{desc}

Candidate A (LLM classification):

Code: {llm_choice['code']}
Title: {llm_choice['title']}

Candidate B (BM25 lexical search):

Code: {bm25_choice['code']}
Title: {bm25_choice['title']}

BM25 score: {bm25_score}

Choose the best UNSPSC code.

Return ONLY the 8 digit code.
"""

    resp = generate_text(prompt)

    match = re.search(r"\b\d{8}\b", resp)

    if match:
        return match.group(0)

    return llm_choice["code"]


# --------------------------------------------------
# TKINTER UI
# --------------------------------------------------

class UNSPSCApp:

    def __init__(self, root):

        self.root = root

        root.title("UNSPSC AI Classifier")

        self.image_path = None
        self.thumbnail = None

        tk.Label(root, text="Description").pack()

        self.input_box = tk.Entry(root, width=80)
        self.input_box.pack()

        tk.Button(root, text="Upload Image", command=self.upload_image).pack()

        self.image_label = tk.Label(root)
        self.image_label.pack()

        tk.Button(root, text="Classify", command=self.run).pack()

        self.log = scrolledtext.ScrolledText(root, height=20)
        self.log.pack()

        self.warning = tk.Label(root, text="", fg="red", font=("Arial", 14, "bold"))
        self.warning.pack()

        self.result = tk.Label(root, text="", font=("Arial", 12, "bold"))
        self.result.pack()

    def upload_image(self):

        path = filedialog.askopenfilename()

        if not path:
            return

        self.image_path = path

        img = Image.open(path)
        img.thumbnail((200, 200))

        self.thumbnail = ImageTk.PhotoImage(img)

        self.image_label.config(image=self.thumbnail)

    def write(self, text):

        self.log.insert(tk.END, text + "\n")
        self.log.see(tk.END)
        self.root.update()

    def run(self):

        desc = self.input_box.get().strip()

        image_desc = None

        if self.image_path:

            self.write("Generating image description...")

            image_desc = describe_image(self.image_path)

            self.write(f"Image description: {image_desc}")

        if not desc and image_desc:

            desc = image_desc
            self.write("Using image description for classification")

        if desc and image_desc:

            match = validate_image_description(desc, image_desc)

            if not match:

                self.warning.config(
                    text="⚠ WARNING: IMAGE DOES NOT MATCH DESCRIPTION"
                )

        self.write("\nExpanding description...")

        item_type, expanded = expand_description(desc)

        self.write(f"Type: {item_type}")
        self.write(f"Expanded: {expanded}")

        tree = services_tree if item_type == "Service" else goods_tree

        llm_code = traverse_tree(expanded, tree, self.write)

        llm_choice = find_details(tree, llm_code)

        bm25 = services_bm25 if item_type == "Service" else goods_bm25
        meta = services_meta if item_type == "Service" else goods_meta

        bm25_choice, score = bm25_search(desc, bm25, meta)

        self.write(f"\nBM25 Candidate: {bm25_choice['code']} Score={score:.2f}")

        final_code = final_decision(expanded, llm_choice, bm25_choice, score)

        final_choice = find_details(tree, final_code)

        self.result.config(
            text=f"FINAL CODE: {final_choice['code']} | {final_choice['title']}"
        )


print("Loading UNSPSC trees...")
goods_tree, services_tree = build_trees(EXCEL_PATH)

print("Building BM25 indices...")
goods_bm25, goods_meta, services_bm25, services_meta = build_bm25(EXCEL_PATH)

root = tk.Tk()
app = UNSPSCApp(root)

root.mainloop()