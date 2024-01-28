"""
    @author: Jibaku789
    @version: 1.1
    @date: January 2023
"""

import gradio as gr
import os
import time
import torch
import numpy as np
import json
import re
import shlex
from PIL import Image

from modules import script_callbacks
from modules import devices, images
from modules.deepbooru import DeepDanbooru

class DeepDanbooruWrapper:

    def __init__(
        self,
    ):

        self.dd_classifier = DeepDanbooru()
        self.cache = {}
        self.enable_cache = False

    def start(self):
        print("Starting DeepDanboru")
        self.dd_classifier.start()

    def stop(self):
        print("Stopping DeepDanboru")
        self.dd_classifier.stop()

    def evaluate_model(self, pil_image, image_id="", minimal_threshold=0):

        if self.enable_cache:
            if image_id and image_id in self.cache:
                return self.cache[image_id]

        # Input image should be 512x512 before reach this point
        pic = images.resize_image(0, pil_image.convert("RGB"), 512, 512)

        #pic = pil_image.convert("RGB")
        a = np.expand_dims(np.array(pic, dtype=np.float32), 0) / 255

        with torch.no_grad(), devices.autocast():
            x = torch.from_numpy(a).to(devices.device)
            y = self.dd_classifier.model(x)[0].detach().cpu().numpy()

        probability_dict = {}
        for tag, probability in zip(self.dd_classifier.model.tags, y):

            if probability < minimal_threshold:
                continue

            if tag.startswith("rating:"):
                continue

            probability_dict[tag] = probability

        if self.enable_cache:
            self.cache[image_id] = probability_dict

        return probability_dict

class DeepDanbooruTag2FolderScript():

    def __init__(self):

        self.source_folder = None
        self.target_folder = None
        self.threshold = None
        self.rules = None
        self.auto_type = None
        self.process_btn = None

        self.csv_info = []
        with open(os.path.join(__file__, "..", "danbooru.csv"), "r", encoding='utf8') as _f:
            for line in _f.readlines():
                self.csv_info.append(shlex.split(line.replace(",", " ").replace("'", "")))

    def on_ui_tabs(self):

        with gr.Blocks(analytics_enabled=False) as ui_component:

            sample = {"<subfolder_name>": {"OR": ["<tag1>", "<tag2>"]}}

            with gr.Row():

                with gr.Column(scale=1, elem_classes="source-image-col"):
                    self.source_folder = gr.Textbox(value="", label="Source Folder", elem_id="deepdanboru_tag2folder_source_folder")
                    self.target_folder = gr.Textbox(value="", label="Target Folder", elem_id="deepdanboru_tag2folder_target_folder")
                    self.auto_type = gr.Dropdown(["None", "Character", "Anime"], value="None", label="Automatic Type", elem_id="deepdanboru_tag2folder_auto_type")
                    self.threshold = gr.Number(value=0.5, label="Threshold", elem_id="deepdanboru_tag2folder_threshold", minimum=0, maximum=1)

                with gr.Column(scale=1, elem_classes="other elements"):
                    self.rules = gr.Textbox(value=json.dumps(sample, indent=True), lines=10, label="Rules", elem_id="deepdanboru_tag2folder_rules_json")

            with gr.Row():
                self.process_btn = gr.Button(value="Process", elem_id="deepdanboru_tag2folder_process_btn")

            self.process_btn.click(
                self.ui_click,
                inputs=[
                    self.source_folder,
                    self.target_folder,
                    self.threshold,
                    self.rules,
                    self.auto_type
                ]
            )

            return [(ui_component, "DeepDanboru Tag2Folder", "deepdanboru_tag2folder_tab")]

    def my_split(self, my_str, token_left="(", token_right=")"):

        matched = []
        tmp = ""
        to_add = False

        for s in my_str:

            if s == token_right:
                to_add = False
                matched.append(tmp)
                tmp = ""

            if to_add:
                tmp += s

            if s == token_left:
                to_add = True

        return matched


    def ui_click(self, source_folder, target_folder, threshold, rules, auto_type):

        json_rules = json.loads(rules)
        source_files = os.listdir(source_folder)
        metrics = {}
        start_time = time.time()

        dd_wrapper = DeepDanbooruWrapper()
        print("Loading DeepDanboru")
        dd_wrapper.start()

        print(f"Start Processing of {len(source_files)} files")
        count = -1
        for _file in source_files:
            count += 1

            if count % 10 == 0:
                print(f"Progress {count}/{len(source_files)}. Elapsed: {time.time() - start_time} segs")

            source_filename = os.path.join(source_folder, _file)

            try:
                with Image.open(source_filename) as pil_image:

                    # Evaluate model
                    model_tags = dd_wrapper.evaluate_model(
                        pil_image,
                        source_filename,
                        threshold
                    )

                    # Move images to new folders
                    found = False
                    for subfolder, conditions in json_rules.items():

                        new_folder = os.path.join(target_folder, subfolder)
                        if not os.path.exists(new_folder):
                            os.makedirs(new_folder)

                        if "AND" in conditions and conditions["AND"]:

                            to_add = True
                            for tag in conditions["AND"]:

                                in_mtag = False
                                for mtag in model_tags:

                                    if "partial:" in tag:
                                        rtag = tag.replace("partial:", "").strip()
                                        if rtag in mtag:
                                            in_mtag = True
                                            break
                                    else:
                                        if tag == mtag:
                                            in_mtag = True
                                            break

                                to_add = to_add and in_mtag

                            if to_add:
                                found = True
                                pil_image.save(os.path.join(new_folder, _file))

                        if "OR" in conditions and conditions["OR"]:

                            to_add = False
                            for tag in conditions["OR"]:

                                in_mtag = False
                                for mtag in model_tags:

                                    if "partial:" in tag:
                                        rtag = tag.replace("partial:", "").strip()
                                        if rtag in mtag:
                                            in_mtag = True
                                            break
                                    else:
                                        if tag == mtag:
                                            in_mtag = True
                                            break

                                to_add = to_add or in_mtag

                            if to_add:
                                found = True
                                pil_image.save(os.path.join(new_folder, _file))


                    if not found:

                        new_folder = os.path.join(target_folder, "unclassified")

                        if auto_type != "None":
                            for tag in model_tags:

                                if auto_type in ["Anime", "Character"]:
                                    model_tag_clean = tag.replace("_", " ")
                                    found_in_csv = False

                                    for csv_tag in self.csv_info:
                                        csv_tag_clean = csv_tag[0].replace("_", " ")
                                        if model_tag_clean == csv_tag_clean:

                                            if csv_tag[1] == "4":

                                                if auto_type == "Character":
                                                    new_folder = os.path.join(target_folder, csv_tag_clean)
                                                    found_in_csv = True
                                                    break

                                                else: #Anime
                                                    if "(" in csv_tag_clean:
                                                        anime_name = self.my_split(" ".join(csv_tag))
                                                        new_folder = os.path.join(target_folder, anime_name[-1])
                                                        found_in_csv = True
                                                        break

                                            if csv_tag[1] == "3" and auto_type == "Anime":
                                                new_folder = os.path.join(target_folder, csv_tag_clean)
                                                found_in_csv = True
                                                break

                                    if found_in_csv:
                                        break

                        if not os.path.exists(new_folder):
                            os.makedirs(new_folder)

                        pil_image.save(os.path.join(new_folder, _file))

                    # Update metrics
                    for tag in model_tags:
                        if tag in metrics:
                            metrics[tag] += 1
                        else:
                            metrics[tag] = 1

            except Exception as e:
                print(e)
                raise

        # Save metrics
        metrics = dict(sorted(metrics.items(), key=lambda x: -x[1]))
        with open(os.path.join(target_folder, "metrics.json"), "w") as _f:
            _f.write(json.dumps(metrics, indent=True))

        dd_wrapper.stop()
        print(f"Finish in {time.time() - start_time} segs")


script = DeepDanbooruTag2FolderScript()
script_callbacks.on_ui_tabs(script.on_ui_tabs)

# end of file
