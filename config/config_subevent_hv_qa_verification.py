"""Configuration loader for subevent human-value QA verification scripts."""

import json
import sys
import os
from openai import api_key

class Config:
    def __init__(self, args):
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
        self.file_name = config["file_name"]
        self.initial_dataset_root_path = config["initial_dataset_root_path"]
        self.process_start = config["process_start"]
        self.process_end = config["process_end"]
        self.file_start = config["file_start"]
        self.file_end = config["file_end"]
        self.incre_file_start = config["incre_file_start"]
        self.incre_file_end = config["incre_file_end"]

        self.prompt_file_path = config["prompt_file_path"]
        self.hv_path = config["hv_path"]
        self.role1 = config["role1"]
        self.content1 = config["content1"]
        self.role2 = config["role2"]
        self.output_root_path = config["output_root_path"]
        self.api_model = config["api_model"]
        self.max_rows_per_file = config["max_rows_per_file"]
        self.min_thread_count = config["min_thread_count"]
        self.subfiles_root_path = config["subfiles_root_path"]
        self.is_process = config["is_process"]
        self.output_filename = config["output_filename"]
        self.completed_filename = config["completed_filename"]
        self.res_filename = config["res_filename"]
        self.log_filename = config["log_filename"]
        self.local = config["local"]
        self.completed_output_root_path = config["completed_output_root_path"]
        self.api_model_hv_eval = config["api_model_hv_eval"]
        self.deepseek_base_url = config["deepseek_base_url"]
        self.output_value_evaluation = config["output_value_evaluation"]
        self.event_base_path = config["event_base_path"]
        self.event_base_incre_path = config["event_base_incre_path"]
        self.output_phase1_v1_root_path = config["output_phase1_v1_root_path"]
        self.output_phase1_v2_root_path = config["output_phase1_v2_root_path"]
        self.phase1_v1_vr_root_path = config["phase1_v1_vr_root_path"]
        self.phase1_v2_vr_root_path = config["phase1_v2_vr_root_path"]

        self.event_base_incre_filename = config["event_base_incre_filename"]
        self.event_base_filename = config["event_base_filename"]

        if self.local == "True":
            self.api_key = os.environ["OPENAI_API_KEY"]
        else:
            self.api_key = args.api_key
        for k, v in args.__dict__.items():
            if v is not None:
                self.__dict__[k] = v

    def __repr__(self):
        return "{}".format(self.__dict__.items())
