import os
import json
import sys
import time
from utils import data_utils

def spent_time(start_time, end_time):
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60
    return hours, minutes, seconds

# compute subfile index by current idx
def compute_subfile_index(idx, start_index, max_rows_per_file):
    current_number = start_index + idx
    file_index = current_number // max_rows_per_file
    return str(file_index)

def str_to_bool(s):
    if s == "True":
        return True
    elif s == "False":
        return False
    else:
        raise ValueError(f"无法转换：{s}")

def add_subevents(datas_subfile, completed_results):
    filtered_datas = []
    for data_subfile in datas_subfile:
        guid_origin = data_subfile["guid"]
        for completed_result in completed_results:
            if completed_result["guid"] == guid_origin:
                sub_events = completed_result["sub_events"]
                data_subfile["sub_events"] = sub_events
                filtered_datas.append(data_subfile)
    return filtered_datas

def add_subevents_hv_eval(datas_subfile, completed_results):
    filtered_datas = []
    for data_subfile in datas_subfile:
        guid_origin = data_subfile["guid"]
        for completed_result in completed_results:
            if completed_result["guid"] == guid_origin:
                if "subevents" not in completed_result.keys():
                    continue
                sub_events = completed_result["subevents"]

                if "argument_mentions" not in completed_result.keys() or "behaviors" not in completed_result.keys() or "story_narratives" not in completed_result.keys():
                    continue
                    # print("not exist")
                argument_mentions = completed_result["argument_mentions"]
                behaviors = completed_result["behaviors"]
                story_narratives = completed_result["story_narratives"]
                # if "subevents" not in data_subfile.keys():
                #     print("test")
                data_subfile["subevents"] = sub_events
                data_subfile["argument_mentions"] = argument_mentions
                data_subfile["behaviors"] = behaviors
                data_subfile["story_narratives"] = story_narratives
                filtered_datas.append(data_subfile)
    return filtered_datas

def add_article_2_comp(datas_subfile, completed_results):
    filtered_datas = []
    for completed_result in completed_results:
        for data_subfile in datas_subfile:
            guid_origin = data_subfile["guid"]
            if completed_result["guid"] == guid_origin:
                if "subevents_human_values" not in completed_result or "story_narrative_human_values" not in completed_result or "behavior_chains_human_values" not in completed_result:
                    # print("subevents_human_values is not exist!")
                    continue
                if "subevents" not in completed_result or "behaviors" not in completed_result or "story_narratives" not in completed_result:
                    print("subevents is not exist!")
                    continue

                completed_result["content"] = data_subfile["content"]
                # completed_result["subevents"] = data_subfile["subevents"]
                # completed_result["argument_mentions"] = data_subfile["argument_mentions"]
                # completed_result["behaviors"] = data_subfile["behaviors"]
                # completed_result["story_narratives"] = data_subfile["story_narratives"]
                filtered_datas.append(completed_result)
    return filtered_datas

def add_subevents_hv_eval_2(datas_subfile, completed_results_2):
    filtered_datas = []
    for data_subfile in datas_subfile:
        guid_origin = data_subfile["guid"]
        flg = False
        for completed_result in completed_results_2:
            if completed_result["guid"] == guid_origin:
                flg = True
        if not flg:
            filtered_datas.append(data_subfile)
    return filtered_datas

def change_hv_info(completed_results_2, completed_results):
    filtered_datas = []
    for completed_result in completed_results:
        for completed_result_2 in completed_results_2:
            guid_2 = completed_result_2["guid"]
            if completed_result["guid"] == guid_2:
                # if "subevents_human_values" not in completed_result or "story_narrative_human_values" not in completed_result or "behavior_chains_human_values" not in completed_result:
                #     # print("subevents_human_values is not exist!")
                #     continue
                # if "subevents" not in completed_result or "behaviors" not in completed_result or "story_narratives" not in completed_result:
                #     print("subevents is not exist!")
                #     continue
                completed_result["article_human_values"] = completed_result_2["article_human_values"]
                completed_result["subevents_human_values"] = completed_result_2["subevents_human_values"]
                completed_result["behavior_chains_human_values"] = completed_result_2["behavior_chains_human_values"]
                completed_result["story_narrative_human_values"] = completed_result_2["story_narrative_human_values"]
                filtered_datas.append(completed_result)

    return filtered_datas

def find_initial(guid, datas_subfile):
    filtered_datas = []
    for data_subfile in datas_subfile:
        if guid == data_subfile["guid"]:
            return data_subfile
    return None

def filter_hv_evaluation(completed_results):
    filtered_datas = []
    error_count = 0
    for completed_result in completed_results:
        keys_to_remove = {'article_human_values', 'subevents_human_values', 'behavior_chains_human_values', 'story_narrative_human_values'}

        # all_keys = [k for k, v in completed_result.items()]
        # if "article_human_values" not in all_keys:
        #     print("item not has the key article_human_values")
        #     error_count = error_count + 1
        #     continue
        # if "subevents_human_values" not in all_keys:
        #     print("item not has the key subevents_human_values")
        #     error_count = error_count + 1
        #     continue
        # if "behavior_chains_human_values" not in all_keys:
        #     print("item not has the key behavior_chains_human_values")
        #     error_count = error_count + 1
        #     continue
        # if "story_narrative_human_values" not in all_keys:
        #     print("item not has the key story_narrative_human_values")
        #     error_count = error_count + 1
        #     continue

        filtered_data = {k: v for k, v in completed_result.items() if k in keys_to_remove}
        filtered_data["guid"] = completed_result["guid"]
        filtered_datas.append(filtered_data)
    return filtered_datas

def filter_event_corpus(completed_results, datas_subfile):
    filtered_datas = []
    for completed_result in completed_results:
        keys_to_remove = {'article_human_values', 'subevents_human_values', 'behavior_chains_human_values', 'story_narrative_human_values'}
        filtered_data = {k: v for k, v in completed_result.items() if k not in keys_to_remove}
        guid = filtered_data["guid"]
        data_subfile = find_initial(guid, datas_subfile)
        filtered_data["content"] = data_subfile["content"]
        filtered_data["time"] = data_subfile["time"]
        filtered_data["title"] = data_subfile["title"]
        filtered_datas.append(filtered_data)
    return filtered_datas

def split_subevent_arguments(subevents):
    subevents_new = []
    argument_mentions_new = []
    keys_to_remove = ['argument_mentions']
    for subevent in subevents:
        subevent_id = subevent["id"]
        if "argument_mentions" not in subevent.keys():
            print("argument_mentions not in subevent")
        argument_mentions = subevent["argument_mentions"]
        argument_mentions["subevent_id"] = subevent_id
        argument_mentions_new.append(argument_mentions)
        subevent = {k: v for k, v in subevent.items() if k not in keys_to_remove}
        subevents_new.append(subevent)
    return argument_mentions_new, subevents_new

def filter_incre_event_corpus(completed_results, datas_subfile):
    filtered_datas = []
    for idx, completed_result in enumerate(completed_results):
        print("current num: "+ str(idx))
        keys_to_remove = {'article_human_values', 'subevents_human_values', 'behavior_chains_human_values', 'story_narrative_human_values'}
        filtered_data = {k: v for k, v in completed_result.items() if k not in keys_to_remove}
        if "argument_mentions" not in completed_result.keys():
            argument_mentions_new, subevents_new = split_subevent_arguments(completed_result["subevents"])
            filtered_data["subevents"] = subevents_new
            filtered_data["argument_mentions"] = argument_mentions_new
        guid = filtered_data["guid"]
        data_subfile = find_initial(guid, datas_subfile)
        filtered_data["content"] = data_subfile["content"]
        filtered_data["time"] = data_subfile["published"]
        filtered_data["title"] = data_subfile["title"]
        filtered_datas.append(filtered_data)
    return filtered_datas

def integrate_event_hv(event_base_dataset, hv_evaluation):
    filtered_datas = []
    for hvs in hv_evaluation:
        for event in event_base_dataset:
            guid_event = event["guid"]
            if hvs["guid"] == guid_event:
                if "subevents_human_values" not in hvs or "story_narrative_human_values" not in hvs or "behavior_chains_human_values" not in hvs:
                    # print("subevents_human_values is not exist!")
                    continue
                if "subevents" not in event or "behaviors" not in event or "story_narratives" not in event:
                    print("subevents is not exist!")
                    continue
                event_cp = event.copy()
                event_cp.update(hvs)
                # event["article_human_values"] = hvs["article_human_values"]
                # event["subevents_human_values"] = hvs["subevents_human_values"]
                # event["behavior_chains_human_values"] = hvs["behavior_chains_human_values"]
                # event["story_narrative_human_values"] = hvs["story_narrative_human_values"]
                filtered_datas.append(event_cp)

    return filtered_datas

def add_actors_to_event_corpus(event_base_data, hv_data):
    filtered_datas = []
    for event_base_data_item in event_base_data:
        guid = event_base_data_item["guid"]
        hv_data_item = find_initial(guid, hv_data)
        if hv_data_item is None:
            print("hv_data_item is None!")
            continue
        actors = hv_data_item["actors"]
        event_base_data_item["actors"] = actors
        filtered_datas.append(event_base_data_item)
    return filtered_datas

def integrate_event_hv(event_base_dataset, hv_evaluation):
    filtered_datas = []
    for hvs in hv_evaluation:
        for event in event_base_dataset:
            guid_event = event["guid"]
            if hvs["guid"] == guid_event:
                if "subevents_human_values" not in hvs or "story_narrative_human_values" not in hvs or "behavior_chains_human_values" not in hvs:
                    # print("subevents_human_values is not exist!")
                    continue
                if "subevents" not in event or "behaviors" not in event or "story_narratives" not in event:
                    print("subevents is not exist!")
                    continue
                event_cp = event.copy()
                event_cp.update(hvs)
                # event["article_human_values"] = hvs["article_human_values"]
                # event["subevents_human_values"] = hvs["subevents_human_values"]
                # event["behavior_chains_human_values"] = hvs["behavior_chains_human_values"]
                # event["story_narrative_human_values"] = hvs["story_narrative_human_values"]
                filtered_datas.append(event_cp)

    return filtered_datas

def story_narr_key_check_p1v1(key, story_narrative):
    if key in story_narrative.keys():
        key_value = story_narrative[key]
        for story_title, story_narr_details in story_narrative.items():
            if story_title != key:
                if isinstance(story_narr_details, str):
                    print("story_narr_details is str")
                if key not in story_narr_details.keys():
                    story_narr_details[key] = key_value
        del story_narrative[key]

def data_format_unification_p1v1(event_content):
    story_narratives = event_content["story_narratives"]
    for story_narrative in story_narratives:
        story_narr_key_check_p1v1("id", story_narrative)
        story_narr_key_check_p1v1("order", story_narrative)
        # print("id in story_narrative!")

def story_narr_key_check_p1v2(key, story_narrative):
    if key in story_narrative.keys():
        key_value = story_narrative[key]
        for story_title, story_narr_details in story_narrative.items():
            if story_title != key:
                if isinstance(story_narr_details, str):
                    print("story_narr_details is str")
                if key not in story_narr_details.keys():
                    story_narr_details[key] = key_value
        del story_narrative[key]

def data_format_unification_p1v2(event_content):
    story_narratives = event_content["story_narratives"]
    for story_narrative in story_narratives:
        story_narr_key_check_p1v2("id", story_narrative)
        story_narr_key_check_p1v2("order", story_narrative)
        # print("id in story_narrative!")

def process_story_narr(event_content):
    story_narratives = event_content["story_narratives"]
    new_story_narratives = []
    for story_narrative in story_narratives:
        new_story_narrative = {}
        title = story_narrative["title"]
        story_narrative_cp = story_narrative.copy()
        del story_narrative_cp["title"]
        new_story_narrative[title] = story_narrative_cp
        new_story_narratives.append(new_story_narrative)

    event_content_cp = event_content.copy()
    del event_content_cp["story_narratives"]
    event_content_cp["story_narratives"] = new_story_narratives
    return event_content_cp

def integrate_processed_data(integrated_dataet, processed_data, exe_type):
    for integrated_data in integrated_dataet:
        guid = integrated_data["guid"]
        related_processed_data = data_utils.finditem(guid, processed_data)
        integrated_data.update(related_processed_data)
    return integrated_dataet

def integrate_eh(voting_results, event_base_dataset, exe_type):
    integrated_dataet = []
    for voting_result in voting_results:
        guid = voting_result["guid"]
        event_content = data_utils.finditem(guid, event_base_dataset)
        if not event_content:
            continue
        print("guid:" + str(guid))
        if guid == "1-15187-1-5":
            print("test!")
        if "content" not in event_content:
            print("test!")
        content = event_content["content"]
        if content == "":
            print("error exist!")
        # data format check
        if exe_type == "p1_v1":
            event_cp = event_content.copy()
            data_format_unification_p1v1(event_cp)
        else:
            event_cp = process_story_narr(event_content)
            data_format_unification_p1v2(event_cp)
        # data format check
        event_cp.update(voting_result)
        # voting_result["content"] = content
        # voting_result["time"] = event_content["time"]
        # voting_result["actors"] = event_content["actors"]
        # voting_result["title"] = event_content["title"]
        integrated_dataet.append(event_cp)
    return integrated_dataet

def orgranize_stream_output_1(stream):
    json_text = ""
    collecting_json = False

    for chunk in stream.text_stream:
        if not collecting_json:
            if "```json" in chunk:
                collecting_json = True
                # 截断掉前面的文字，只保留```json之后的内容
                json_text += chunk.split("```json")[-1]
        elif "```" in chunk:
            # 到了结尾，去除```并结束收集
            json_text += chunk.split("```")[0]
            break
        else:
            json_text += chunk
    return json_text


def orgranize_stream_output_1(stream):
    full_text = ""
    for chunk in stream.text_stream:
        full_text += chunk

    # 尝试从最后一个 { 开始，到最后一个 } 结束
    start = full_text.find("{")
    end = full_text.rfind("}")

    if start != -1 and end != -1 and end > start:
        json_str = full_text[start:end + 1]
        return json_str
    else:
        return full_text

def hv_importance_evaluation(conflict_count, human_value_type, subevent_id, story_narr_id, behavior_chain_ids, related_subevent_ids, source_human_values, hv_type, related_mappings, impact_result):
    hv_dicts = []

    for actor, human_value_lists in source_human_values.items():
        if human_value_type == "subevents_human_values":
            actor_importance = data_utils.compute_actor_importance(related_mappings)
        elif human_value_type == "story_narrative_human_values":
            actor_importance = data_utils.compute_actor_importance(related_mappings)
        elif human_value_type == "behavior_chains_human_values":
            actor_importance = data_utils.compute_actor_importance(related_mappings)
        elif human_value_type == "article_human_values":
            actor_importance = data_utils.compute_actor_importance(related_mappings)
        else:
            actor_importance[actor] = None

        for hv_detail in human_value_lists:
            for hv_label, hv_desc in hv_detail.items():
                if actor not in actor_importance:
                    print("test")
                hv_desc["actor_importance"] = actor_importance[actor]
                hv_importance = data_utils.compute_hv_importance(impact_result)
                if (actor, hv_label) not in hv_importance:
                    print("error")
                hv_desc["hv_importance"] = hv_importance[(actor, hv_label)]



                # hv_dict = {}
                # if "confidence" not in hv_desc.keys():
                #     print("confidence not exist!")
                # hv_dict["actor"] = actor
                # hv_dict["human_value_label"] = hv_label
                # hv_dict["id"] = conflict_count["count"]
                # hv_dicts.append(hv_dict)
                # conflict_count["count"] += 1
    return hv_dicts


def behavior_chains_importance_integration(human_value_type, behavior_chains_human_values, behavior_chains_impact_result, mapping_results, subevents, behavior_chains, direction_types):
    behavior_chains_verification_questions = {}
    behavior_chains_hvs = []
    for behavior_chain_human_values in behavior_chains_human_values:
        if behavior_chain_human_values:
            # 根据 behavior 查询相关的子事件，子事件 id，行为链
            if "behavior_ids_chains" not in behavior_chain_human_values:
                print("test!")
            behavior_ids_chains = behavior_chain_human_values["behavior_ids_chains"]
            behavior_chain, chain_type = data_utils.find_behavior_chains(behavior_ids_chains, behavior_chains)
            related_subevent_ids = data_utils.find_behavior_related_subevent_ids(behavior_chain, chain_type)
            related_subevents = data_utils.find_subevents_with_id(related_subevent_ids, subevents)
            related_mappings = data_utils.find_related_mappings(related_subevent_ids, mapping_results)
            related_impact_result = data_utils.find_item_by_keys(behavior_ids_chains, "behavior_ids_chains", behavior_chains_impact_result)
            # ==========================================
            behavior_chains_hv_values = {}
            aligned_count = {"count": 0}
            contradictory_count = {"count": 0}
            if direction_types[0] in behavior_chain_human_values:
                aligned_with_human_values = behavior_chain_human_values[direction_types[0]]
                aligned_with_impact_result = related_impact_result[direction_types[0]]
                if aligned_with_human_values:
                    aligned_hvs = hv_importance_evaluation(aligned_count, human_value_type, "", [], behavior_ids_chains,
                                                         related_subevent_ids,
                                                         aligned_with_human_values, "aligned",
                                                         related_mappings, aligned_with_impact_result)
                    behavior_chains_hv_values["aligned_with_human_values"] = aligned_hvs

            if direction_types[1] in behavior_chain_human_values:
                contradictory_to_human_values = behavior_chain_human_values[direction_types[1]]
                contradictory_impact_result = related_impact_result[direction_types[1]]
                if contradictory_to_human_values:
                    contradictory_hvs = hv_importance_evaluation(contradictory_count, human_value_type, "", [], behavior_ids_chains, related_subevent_ids, contradictory_to_human_values, "contradictory", related_mappings, contradictory_impact_result)
                    behavior_chains_hv_values["contradictory_to_human_values"] = contradictory_hvs
            if behavior_chains_hv_values:
                behavior_chains_hv_values["behavior_ids_chains"] = behavior_ids_chains
                behavior_chains_hv_values["behavior_chains"] = behavior_chain
                behavior_chains_hv_values["related_subevents"] = related_subevents
                behavior_chains_hvs.append(behavior_chains_hv_values)

    if behavior_chains_hvs:
        behavior_chains_verification_questions["behavior_chains_verification_questions"] = behavior_chains_hvs
    return behavior_chains_verification_questions

def temp_article_importance_integration(human_value_type, temp_article_human_values, temp_article_impact_result, mapping_results, direction_types):
    article_verification_questions = {}
    article_hv_values = {}
    aligned_conflict_count = {"count": 0}
    contradictory_conflict_count = {"count": 0}

    if direction_types[0] in temp_article_human_values:
        aligned_with_human_values = temp_article_human_values[direction_types[0]]
        aligned_with_impact_result = temp_article_impact_result[direction_types[0]]
        if aligned_with_human_values:
            aligned_hvs = hv_importance_evaluation(aligned_conflict_count, human_value_type, "", "", [], [],
                                                 aligned_with_human_values, direction_types[0],
                                                 mapping_results, aligned_with_impact_result)
            article_hv_values[direction_types[0]] = aligned_hvs

    if direction_types[1] in temp_article_human_values:
        contradictory_to_human_values = temp_article_human_values[direction_types[1]]
        contradictory_impact_result = temp_article_impact_result[direction_types[1]]
        if contradictory_to_human_values:
            contradictory_hvs = hv_importance_evaluation(contradictory_conflict_count, human_value_type, "", "", [], [],
                                                       contradictory_to_human_values,
                                                       direction_types[1], mapping_results, contradictory_impact_result)
            article_hv_values[direction_types[1]] = contradictory_hvs

    if article_hv_values:
        article_verification_questions["article_verification_questions"] = article_hv_values
    return article_verification_questions

def story_narr_importance_integration(human_value_type, story_narrs_human_values, story_impact_result, mapping_results, subevents, story_narratives, direction_types):
    story_narrs_verification_questions = {}
    story_narr_hvs = []
    for story_narr_human_values in story_narrs_human_values:
        if story_narr_human_values:
            story_narr_id = data_utils.find_hv_specific_id("story_narrative_id", story_narr_human_values)
            if story_narratives == None:
                print("story_narratives not exist!")
            story_narrative = data_utils.find_story_narrative(story_narr_id, story_narratives)
            if story_narrative == None:
                print("story_narratives not exist!")
            related_subevent_ids = data_utils.find_related_subevent_ids(story_narrative)
            if related_subevent_ids == None:
                print("related_subevent_ids not exist!")
            related_subevents = data_utils.find_subevents_with_id(related_subevent_ids, subevents)
            related_mappings = data_utils.find_related_mappings(related_subevent_ids, mapping_results)
            related_impact_result = data_utils.find_item_by_keys(story_narr_id, "story_narrative_id",
                                                                story_impact_result)

            story_narr_hv_values = {}
            aligned_count = {"count": 0}
            contradictory_count = {"count": 0}
            if direction_types[0] in story_narr_human_values:
                aligned_with_human_values = story_narr_human_values[direction_types[0]]
                aligned_with_impact_result = related_impact_result[direction_types[0]]
                if aligned_with_human_values:
                    aligned_hvs = hv_importance_evaluation(aligned_count, human_value_type, "", story_narr_id, [], related_subevent_ids,
                                                         aligned_with_human_values,
                                                         "aligned", related_mappings, aligned_with_impact_result)
                    story_narr_hv_values["aligned_with_human_values"] = aligned_hvs

            if direction_types[1] in story_narr_human_values:
                contradictory_to_human_values = story_narr_human_values[direction_types[1]]
                contradictory_impact_result = related_impact_result[direction_types[1]]
                if contradictory_to_human_values:
                    contradictory_hvs = hv_importance_evaluation(contradictory_count, human_value_type, "", story_narr_id, [], related_subevent_ids, contradictory_to_human_values, "contradictory", related_mappings, contradictory_impact_result)
                    story_narr_hv_values["contradictory_to_human_values"] = contradictory_hvs
            if story_narr_hv_values:
                story_narr_hv_values["story_narrative_id"] = story_narr_id
                story_narr_hv_values["story_narrative"] = story_narrative
                story_narr_hv_values["related_subevents"] = related_subevents
                story_narr_hvs.append(story_narr_hv_values)
    if story_narr_hvs:
        story_narrs_verification_questions["story_narrative_verification_questions"] = story_narr_hvs
    return story_narrs_verification_questions

def subevents_importance_integration(human_value_type, subevents_human_values, subevent_impact_result, mapping_results, subevents, direction_types, sent_subevent_mapping_results):
    subevents_verification_questions = {}
    subevent_hvs = []
    for subevent_human_values in subevents_human_values:
        subevent_id = data_utils.find_subevent_id(subevent_human_values)
        subevent = data_utils.find_subevent(subevent_id, subevents)
        sentence_ids = data_utils.find_sentence_ids(subevent_id, subevents)
        related_mappings = data_utils.find_related_mappings(sentence_ids, sent_subevent_mapping_results)
        related_impact_result = data_utils.find_item_by_keys(subevent_id, "subevent_id",
                                                            subevent_impact_result)
        sub_event_hv_values = {}
        aligned_count = {"count": 0}
        contradictory_count = {"count": 0}
        if direction_types[0] in subevent_human_values:
            aligned_with_human_values = subevent_human_values[direction_types[0]]
            aligned_with_impact_result = related_impact_result[direction_types[0]]
            if aligned_with_human_values:
                aligned_hvs = hv_importance_evaluation(aligned_count, human_value_type, subevent_id, "", [], [],
                                                     aligned_with_human_values, "aligned",
                                                     related_mappings, aligned_with_impact_result)
                sub_event_hv_values["aligned_with_human_values"] = aligned_hvs

        if direction_types[1] in subevent_human_values:
            contradictory_to_human_values = subevent_human_values[direction_types[1]]
            contradictory_impact_result = related_impact_result[direction_types[1]]
            if contradictory_to_human_values:
                contradictory_hvs = hv_importance_evaluation(contradictory_count, human_value_type, subevent_id, "", [], [], contradictory_to_human_values, "contradictory", related_mappings, contradictory_impact_result)
                sub_event_hv_values["contradictory_to_human_values"] = contradictory_hvs

        if sub_event_hv_values:
            sub_event_hv_values["subevent_id"] = subevent_id
            sub_event_hv_values["subevent"] = subevent["subevent"]
        if sub_event_hv_values:
            subevent_hvs.append(sub_event_hv_values)

    if subevent_hvs:
        subevents_verification_questions["subevents_verification_questions"] = subevent_hvs
    return subevents_verification_questions

def hv_importance_integration(hv_types, hv_removal_types, datas_subfiles, direction_types):
    prompt_cons_file = {}
    for filename, data_list in datas_subfiles.items():
        print("========================" + "filename:" + str(filename) + "========================")
        prompt_cons_subfile = []
        for paragraph in data_list:
            # =============== test ================
            if paragraph["guid"] == "2-34484-7-t":
                print("find data!")
            # =============== test ================
            prompt_content_map = {}
            subevents = paragraph["subevents"]
            if hv_removal_types[0] in paragraph:
                temp_article_impact_result = paragraph[hv_removal_types[0]]["assessment_results"]
            else:
                temp_article_impact_result = []
            if hv_removal_types[1] in paragraph:
                subevent_impact_result = paragraph[hv_removal_types[1]]["assessment_results"]
            else:
                subevent_impact_result = []
            if hv_removal_types[2] in paragraph:
                behavior_impact_result = paragraph[hv_removal_types[2]]["assessment_results"]
            else:
                behavior_impact_result = []
            if hv_removal_types[3] in paragraph:
                story_impact_result = paragraph[hv_removal_types[3]]["assessment_results"]
            else:
                story_impact_result = []
            mapping_results = paragraph["mapping_results"]
            sent_subevent_mapping_results = paragraph["mappings"]

            # if behavior_impact_result:
            #     behaviors = paragraph["behaviors"]
            #     behavior_chains_human_values = paragraph["behavior_chains_human_values"]
            #     behavior_chains_verification_questions = behavior_chains_importance_integration(hv_types[2], behavior_chains_human_values, behavior_impact_result, mapping_results,
            #                                                                 subevents, behaviors, direction_types)
            #     if not behavior_chains_verification_questions:
            #         continue
            #     behavior_chains_verification_questions_text = json.dumps(behavior_chains_verification_questions,
            #                                                             ensure_ascii=False)
            # if temp_article_impact_result:
            #     article_human_values = paragraph["article_human_values"]
            #     article_verification_questions = temp_article_importance_integration(hv_types[0],
            #                                                                       article_human_values, temp_article_impact_result, mapping_results, direction_types)
            #     if not article_verification_questions:
            #         continue
            #     article_verification_questions_text = json.dumps(article_verification_questions,
            #                                                              ensure_ascii=False)
            #
            # if story_impact_result:
            #     story_narratives = paragraph["story_narratives"]
            #     story_narrative_human_values = paragraph["story_narrative_human_values"]
            #     if story_narratives == None:
            #         print("story_narratives not exist!")
            #     story_narstory_verification_questions = story_narr_importance_integration(hv_types[3], story_narrative_human_values, story_impact_result, mapping_results, subevents, story_narratives, direction_types)
            #     if not story_narstory_verification_questions:
            #         continue
            #     story_narstory_verification_questions_text = json.dumps(story_narstory_verification_questions, ensure_ascii=False)
            if subevent_impact_result:
                subevent_human_values = paragraph["subevents_human_values"]
                subevents_verification_questions = subevents_importance_integration(hv_types[1], subevent_human_values, subevent_impact_result, mapping_results, subevents, direction_types, sent_subevent_mapping_results)
                if not subevents_verification_questions:
                    continue
                subevents_verification_questions_text = json.dumps(subevents_verification_questions, ensure_ascii=False)

            prompt_cons_subfile.append(prompt_content_map)
        if prompt_cons_subfile:
            prompt_cons_file[filename] = prompt_cons_subfile
    return prompt_cons_file