import argparse
import random
import numpy as np
import time
import json
import os
# from modelscope.msdatasets import MsDataset
import pickle


def find_node(s_guid):
    new_ds = []
    node_index_file = open("dataset/processed_data/node_index.pickle", 'rb')
    node_index_ds = pickle.load(node_index_file)
    # guids_97 = node_index_ds[97]["guid"]
    # for guid_97 in guids_97:
    for node_index_d in node_index_ds.values():
        guid = node_index_d["guid"]
        if len(guid) == 1 and s_guid == guid[0]:
            new_ds.append(node_index_d)
    return new_ds


def find_cv_cap(s_guid):
    new_ds = []
    for i in range(36):
        pre_cv_process_cap_file = open("UKnow/cv_cap/pre_cv_process_cap" + i + ".pickle", 'rb')
        pre_cv_process_cap_ds = pickle.load(pre_cv_process_cap_file)
        # guids_97 = node_index_ds[97]["guid"]
        # for guid_97 in guids_97:
        for node_index_d in pre_cv_process_cap_ds.values():
            guid = node_index_d["guid"]
            if len(guid) == 1 and s_guid == guid[0]:
                new_ds.append(node_index_d)
        return new_ds


if __name__ == '__main__':

    pre_node_file = open("UKnow/processed_data/pre_node.pickle", 'rb')
    pre_node_ds = pickle.load(pre_node_file)
    print("abc")

    for guid in pre_node_ds.keys():
        pre_node_value = pre_node_ds[guid]
        imgpth = pre_node_value["imgpth"]
        time = pre_node_value["time"]
        title = pre_node_value["title"]
        content = pre_node_value["content"]
        imgdes = pre_node_value["imgdes"]
        title_nlpner = pre_node_value["title_nlpner"]
        content_nlpner = pre_node_value["content_nlpner"]

        if pre_node_value.__contains__("img_cvdetbbox"):
            img_cvdetbbox = pre_node_value["img_cvdetbbox"]
        if pre_node_value.__contains__("img_cvobj"):
            img_cvobj = pre_node_value["img_cvobj"]
        if pre_node_value.__contains__("img_cap"):
            img_cap = pre_node_value["img_cap"]

        print("abc")
    # for pre_node_d in pre_node_ds:
    #     print(pre_node_d)
