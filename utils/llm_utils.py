import os
import json
import sys
import time
from openai import OpenAI
from google import genai
from google.genai import types
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from utils import preprocess_utils


# os.environ["OPENAI_API_KEY"]
def open_api_instantiate(api_key):
    openapi_client = OpenAI(
        api_key=api_key
    )
    return openapi_client

def anthropic_instantiate(api_key):
    anthropic_client = Anthropic(
        api_key=api_key
    )
    return anthropic_client

def open_api_instantiate_deepseek(api_key, base_url):
    openapi_client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    return openapi_client

def call_open_api(api_model, openapi_client, api_role1, api_content1, api_role2, api_content2):
    if "deepseek" in api_model:
        response_completion = openapi_client.chat.completions.create(
            model=api_model,
            messages=[
                {
                    "role": api_role1,
                    "content": api_content1
                },
                {
                    "role": api_role2,
                    "content": api_content2
                }
            ],
            # extra_body={"thinking": {"type": "enabled"}}
            top_p=1.0,
            response_format={"type": "json_object"}
        )
    else:
        response_completion = openapi_client.chat.completions.create(
            model=api_model,
            messages=[
                {
                    "role": api_role1,
                    "content": api_content1
                },
                {
                    "role": api_role2,
                    "content": api_content2
                }
            ],
            # temperature=0.0,
            top_p=1.0,
            response_format={"type": "json_object"}
        )
    return response_completion

def open_api_instantiate_deepseek(api_key, base_url):
    openapi_client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    return openapi_client

def call_open_api_deepresearch(api_model, openapi_client, api_role1, api_content1, api_role2, api_content2):
    response_completion = openapi_client.responses.create(
        model=api_model,
        input=[
            {
                "role": api_role1,
                "content": [
                    {
                        "type": "input_text",
                        "text": api_content1
                    }
                ]
            },
            {
                "role": api_role2,
                "content": [
                    {
                        "type": "input_text",
                        "text": api_content2
                    }
                ]
            }
        ],
        # reasoning={
        #     "summary": "auto"
        # },
        tools=[
            {
                "type": "web_search_preview"
            }
        ]
    )
    return response_completion

def call_open_api_o1_mini_preview(api_model, openapi_client, api_role1, api_content1, api_role2, api_content2):
    response_completion = openapi_client.chat.completions.create(
        model=api_model,
        messages=[
            {
                "role": api_role2,
                "content": api_content2
            }
        ],
        max_completion_tokens=32768
    )
    return response_completion


def call_open_api_o3_mini(api_model, openapi_client, api_role1, api_content1, api_role2, api_content2):
    response_completion = openapi_client.chat.completions.create(
        model=api_model,
        messages=[
            {
                "role": api_role2,
                "type": "text",
                "content": api_content2
            }
        ]
    )
    return response_completion

def sensitive_check(client, content):
    response = client.moderations.create(
      model="omni-moderation-latest",
      input=content,
    )
    return response

def call_google_gemini(api_model, client, api_role1, api_content1, api_role2, api_content2):
    # client = genai.Client(api_key="GEMINI_API_KEY")
    # print(response.text)
    response = client.models.generate_content(
        model=api_model,
        # contents=[image, api_content2],
        contents=api_content2,
        config=types.GenerateContentConfig(
            temperature=0.1
        )
    )
    return response

# call claude api
def call_anthropic(api_model, client, api_role1, api_content1, api_role2, api_content2):
    response = client.messages.create(
        model=api_model,
        max_tokens=32000,
        temperature=1,
        system=api_content1,
        messages=[
            {
                "role": api_role2,
                "content": [
                    {
                        "type": "text",
                        "text": api_content2
                    }
                ]
            }
        ]
    )
    return response

def orgranize_stream_output(stream):
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

def call_anthropic_stream(api_model, client, api_role1, api_content1, api_role2, api_content2):
    json_text = ""
    collecting_json = False
    with client.messages.stream(
            model=api_model,
            max_tokens=32000,
            messages=[{"role": api_role2, "content": api_content2}],
    ) as stream:
        # for text in stream.text_stream:
        #     print(text, end="", flush=True)
        json_text = preprocess_utils.orgranize_stream_output_1(stream)
    return json_text

# ----------------------------
# G1 API inference (skeleton)
# You need to fill in your own client calls.
# ----------------------------
def api_call_stub(provider: str, model_name: str, prompt: str, api_key: str) -> str:
    """
    Replace this stub with your own OpenAI/Anthropic/Gemini/DeepSeek SDK calls.
    Must return model output text.
    """
    raise NotImplementedError(
        f"API call not implemented for provider={provider}. "
        f"Please plug in your SDK call here."
    )
