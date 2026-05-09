import os
import json
import sys
import time


def hv_count_statistics(hv_type, human_values, count_common_aligned_with_human_values
                                , count_conflict_aligned_with_human_values, count_common_contradictory_to_human_values
                                , count_conflict_contradictory_to_human_values):
    for actor, human_value in human_values.items():
        if actor == "subevent_id" or actor == "story_narrative_id" or actor == "behavior_ids_chains":
            continue
        common_hvs_align = human_value["common_aligned_with_human_values"]
        conflict_hvs_align = human_value["conflict_aligned_with_human_values"]
        common_hvs_contra = human_value["common_contradictory_to_human_values"]
        conflict_hvs_contra = human_value["conflict_contradictory_to_human_values"]
        if hv_type not in count_common_aligned_with_human_values:
            count_common_aligned_with_human_values[hv_type] = 1
        else:
            count_common_aligned_with_human_values[hv_type] += len(common_hvs_align)

        if hv_type not in count_conflict_aligned_with_human_values:
            count_conflict_aligned_with_human_values[hv_type] = 1
        else:
            count_conflict_aligned_with_human_values[hv_type] += len(conflict_hvs_align)

        if hv_type not in count_common_contradictory_to_human_values:
            count_common_contradictory_to_human_values[hv_type] = 1
        else:
            count_common_contradictory_to_human_values[hv_type] += len(common_hvs_contra)

        if hv_type not in count_conflict_contradictory_to_human_values:
            count_conflict_contradictory_to_human_values[hv_type] = 1
        else:
            count_conflict_contradictory_to_human_values[hv_type] += len(conflict_hvs_contra)

def hvs_count_statistics(hv_type, human_values, count_common_aligned_with_human_values
                                , count_conflict_aligned_with_human_values, count_common_contradictory_to_human_values
                                , count_conflict_contradictory_to_human_values):
    if hv_type == "article_human_values":
        hv_count_statistics(hv_type, human_values, count_common_aligned_with_human_values
                                , count_conflict_aligned_with_human_values, count_common_contradictory_to_human_values
                                , count_conflict_contradictory_to_human_values)
    else:
        for human_value_item in human_values:
            hv_count_statistics(hv_type, human_value_item, count_common_aligned_with_human_values
                                , count_conflict_aligned_with_human_values, count_common_contradictory_to_human_values
                                , count_conflict_contradictory_to_human_values)

def common_conflict_statistics(total_integrated_data_p1_v1):
    count_common_aligned_with_human_values = {}
    count_conflict_aligned_with_human_values = {}
    count_common_contradictory_to_human_values = {}
    count_conflict_contradictory_to_human_values = {}

    for filename, hv_data in total_integrated_data_p1_v1.items():
        for item in hv_data:
            subevents_human_values = item["subevents_human_values"]
            # article_human_values = item["article_human_values"]
            # behavior_chains_human_values = item["behavior_chains_human_values"]
            # story_narrative_human_values = item["story_narrative_human_values"]
            hvs_count_statistics("subevents_human_values", subevents_human_values, count_common_aligned_with_human_values
                                , count_conflict_aligned_with_human_values, count_common_contradictory_to_human_values
                                , count_conflict_contradictory_to_human_values
                                )

            # hvs_count_statistics("article_human_values", article_human_values, count_common_aligned_with_human_values
            #                     , count_conflict_aligned_with_human_values, count_common_contradictory_to_human_values
            #                     , count_conflict_contradictory_to_human_values
            #                     )
            #
            # hvs_count_statistics("behavior_chains_human_values", behavior_chains_human_values, count_common_aligned_with_human_values
            #                     , count_conflict_aligned_with_human_values, count_common_contradictory_to_human_values
            #                     , count_conflict_contradictory_to_human_values
            #                     )
            #
            # hvs_count_statistics("story_narrative_human_values", story_narrative_human_values, count_common_aligned_with_human_values
            #                     , count_conflict_aligned_with_human_values, count_common_contradictory_to_human_values
            #                     , count_conflict_contradictory_to_human_values
            #                     )

    print("count_common_aligned_with_human_values: " + str(count_common_aligned_with_human_values))
    print("count_conflict_aligned_with_human_values: " + str(count_conflict_aligned_with_human_values))
    print("count_common_contradictory_to_human_values: " + str(count_common_contradictory_to_human_values))
    print("count_conflict_contradictory_to_human_values: " + str(count_conflict_contradictory_to_human_values))
    return (count_common_aligned_with_human_values, count_conflict_aligned_with_human_values,
            count_common_contradictory_to_human_values, count_conflict_contradictory_to_human_values)

def politics_article_statistics(total_integrated_data_p1_v1):
    politics_article_file = {}
    event_categories = {}
    communist = 0
    for filename, hv_data in total_integrated_data_p1_v1.items():
        politics_article = []
        for item in hv_data:
            subevents = item["subevents"]
            content = item["content"]
            politics_flag = False
            for subevent in subevents:
                subevent_genre = subevent["subevent_genre"]
                if subevent_genre.startswith("PoliticalEvents"):
                    politics_flag = True
                if subevent_genre not in event_categories.keys():
                    event_categories[subevent_genre] = 1
                else:
                    event_categories[subevent_genre] += 1
            if politics_flag:
                politics_article.append(item)
            if "communist party" in content:
                communist += 1

        if politics_article:
            politics_article_file[filename] = politics_article

    politics_categories = []
    for event_category, count in event_categories.items():
        if event_category.startswith("PoliticalEvents"):
            politics_categories.append((event_category, count))

    return (politics_article_file, politics_categories)