import os
import json
import sys
import time



def data_integration_check(all_datas_subfiles):
    p1v1_datas_gpt_41 = all_datas_subfiles["phase1_v1"]["gpt-4.1"]
    p1v1_datas_gpt_4o = all_datas_subfiles["phase1_v1"]["gpt-4o"]
    p1v1_datas_gemini_25pro = all_datas_subfiles["phase1_v1"]["gemini-2.5-pro"]

    p1v2_all_datas_subfiles["phase1_v2"]["gpt-4.1"]
    p1v2_all_datas_subfiles["phase1_v2"]["gpt-4o"]
    p1v2_all_datas_subfiles["phase1_v2"]["gemini-2.5-pro"]


def check_hv_type_exist(check_key, item_m1, item_m2, item_m3):
    if check_key in item_m1 and len(item_m1[check_key]) > 0:
        if check_key in item_m2 and len(item_m2[check_key]) > 0:
            if check_key in item_m3 and len(item_m3[check_key]) > 0:
                return True
            else:
                return False
        else:
            return False
    else:
        return False