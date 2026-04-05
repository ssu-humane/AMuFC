from openai import OpenAI
from PIL import Image
from io import BytesIO
import base64
import time
from dotenv import load_dotenv
import os
from google import genai
from google.genai import types

import torch
import numpy as np
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import Qwen2VLForConditionalGeneration, MllamaForConditionalGeneration, AutoProcessor, AutoModel, AutoTokenizer

load_dotenv()

BEFORECHECK_PROMPT = """Your task is to determine whether the image evidence is essential to verify the given claim or clarify the provided text evidence. To do this, follow these steps:
1. Analyze the claim and the text evidence to understand the context.
2. Evaluate if verifying the claim or identifying the text evidence requires additional visual context.
3. Decide if the image evidence is indispensable for verification or clarification.
Respond only with 'Yes' if the image evidence is necessary or 'No' if it is not."""

AFTERCHECK_PROMPT = """Your task is to determine if the provided image evidence is essential to verify the given claim or clarify the provided text evidence. To do this, follow these steps:
1. Analyze the claim and the text evidence to understand the context.
2. Assess whether the image evidence provides critical information not conveyed by the text alone.
3. Decide if the image evidence is necessary for verification or clarification.
Respond only with 'Yes' if the image evidence is necessary or 'No' if it is not."""

ANALYZE_PROMPT = """Your task is to determine whether the provided image evidence is necessary for verifying the given claim or clarifying the accompanying text evidence. Follow these steps:
1. Analyze the claim and the text evidence to understand the context.
2. Assess whether the image provides important information that is not already conveyed by the text.
3. Decide whether the image is necessary for verification and justify your reasoning.
Respond only with your analysis."""

ANALYZE_FACTCHECK_PROMPT = "Your task is to predict the verdict of a claim based on the provided image evidence with analysis and text evidence. The possible labels are: 'supported', 'refuted', 'NEI'. NEI stands for Not Enough Information. Respond only with the correct label."

MULTIMODAL_FACTCHECK_PROMPT = "Your task is to predict the verdict of a claim based on the provided image evidence and text evidence. The possible labels are: 'supported', 'refuted', 'NEI'. NEI stands for Not Enough Information. Respond only with the correct label."

TEXT_FACTCHECK_PROMPT = "Your task is to predict the verdict of a claim based on the provided text evidence. The possible labels are: 'supported', 'refuted', 'NEI'. NEI stands for Not Enough Information. Respond only with the correct label."

CLAIM_FACTCHECK_PROMPT = "Your task is to predict the verdict of a claim. The possible labels are: 'supported', 'refuted', 'NEI'. NEI stands for Not Enough Information. Respond only with the correct label."

MULTIMODAL_JUSTIFICATION_PROMPT = "Your task is to generate a brief justification of a verdict based on a claim with image and text evidence. Respond only with your justification."

TEXT_JUSTIFICATION_PROMPT = "Your task is to generate a brief justification of a verdict based on a claim with text evidence. Respond only with your justification."

def check_label(output_text: str) -> str:
    # make all lowercase for easier comparison
    output_text = output_text.lower()

    if output_text=="supported" or "supported" in output_text:
        return "supported"
    elif output_text=="true" or "true" in output_text:
        return "supported"
    elif output_text=="refuted" or "refuted" in output_text:
        return "refuted"
    elif output_text=="false" or "false" in output_text:
        return "refuted"
    elif output_text=="nei" or "nei" in output_text:
        return "NEI"
    elif output_text=="not enough information" or "not enough information" in output_text:
        return "NEI"
    else:
        return None

def check_use_image(output_text: str) -> bool:
    # make all lowercase for easier comparison
    output_text = output_text.lower()

    if output_text=="no" or "no" in output_text:
        return "No"
    elif output_text=="yes" or "yes" in output_text:
        return "Yes"
    else:
        return None

class openai_model():
    '''
    Base class for model compatibility implementation.
    '''
    def __init__(self, model_name) -> None:
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.MODEL = model_name
   
    # TODO: Add regenerating logic to the generate methods

    def return_prompt(self, type) -> str:
        if type == "beforecheck":
            return BEFORECHECK_PROMPT
        elif type == "aftercheck":
            return AFTERCHECK_PROMPT
        elif type == "multimodal":
            return MULTIMODAL_FACTCHECK_PROMPT
        elif type == "text":
            return TEXT_FACTCHECK_PROMPT
        elif type == "claim":
            return CLAIM_FACTCHECK_PROMPT
        elif type == "analyze":
            return ANALYZE_PROMPT
        elif type == "analyze_factcheck":
            return ANALYZE_FACTCHECK_PROMPT
        elif type == "justification":
            return MULTIMODAL_JUSTIFICATION_PROMPT + TEXT_JUSTIFICATION_PROMPT
        else:
            raise ValueError("Invalid prompt type")

    def encode_image(self, image_path) -> str:
        if type(image_path) == str:
            # Open the image
            img = Image.open(image_path)

            # Resize the image if its width or height is greater than 512
            if img.width > 512 or img.height > 512:
                img.thumbnail((512, 512))

            if img.width < 256 or img.height < 256:
                new_w = max(img.width, 256)
                new_h = max(img.height, 256)
                img = img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)

            buffered = BytesIO()
            img.save(buffered, format="PNG")
            buffered.seek(0)

            # Return base64-encoded string of the resized image
            return base64.b64encode(buffered.read()).decode("utf-8")
        else:
            image_list = []
            for path in image_path:
                # Open the image
                img = Image.open(path)

                # Resize the image if its width or height is greater than 512
                if img.width > 512 or img.height > 512:
                    img.thumbnail((512, 512))

                if img.width < 256 or img.height < 256:
                    new_w = max(img.width, 256)
                    new_h = max(img.height, 256)
                    img = img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)

                buffered = BytesIO()
                img.save(buffered, format="PNG")
                buffered.seek(0)
                # Append base64-encoded string of the resized image to the list
                image_list.append(base64.b64encode(buffered.read()).decode("utf-8"))
            return image_list

    def generate_beforecheck(self, data) -> str:
        claim = "\nClaim: " + data["claim"]
        text_evidence = "\nText Evidence: " + data["evidence"]

        try:
            attempt = 0
            while attempt < 3:
                response = self.client.chat.completions.create(
                    model=self.MODEL,
                    messages=[
                        {"role": "system", "content": BEFORECHECK_PROMPT},
                        {"role": "user", "content": [
                            {"type": "text", "text": claim + text_evidence},
                        ]}
                    ],
                    temperature=0.0,
                )

                output_text = response.choices[0].message.content

                if check_use_image(output_text) is None:
                    attempt += 1
                    time.sleep(1)
                else:
                    label = check_use_image(output_text)
                    break

            if attempt == 3:
                label = "Yes"

        except Exception as e:
            # print error message
            print(e)
            output_text = "NULL Response"
            label = "Yes"
        return output_text, label

    def generate_aftercheck(self, data) -> str:
        img = self.encode_image(data['image'])
        claim = "\nClaim: " + data["claim"]
        image_evidence = "\nImage Evidence: "
        text_evidence = "\nText Evidence: " + data["evidence"]

        try:
            attempt = 0
            while attempt < 3:
                response = self.client.chat.completions.create(
                    model=self.MODEL,
                    messages=[
                        {"role": "system", "content": AFTERCHECK_PROMPT},
                        {"role": "user", "content": [
                            {"type": "text", "text": claim + image_evidence},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/png;base64,{img}"}
                            },
                            {"type": "text", "text": text_evidence}
                        ]}
                    ],
                    temperature=0.0,
                )

                output_text = response.choices[0].message.content

                if check_use_image(output_text) is None:
                    attempt += 1
                    time.sleep(1)
                else:
                    label = check_use_image(output_text)
                    break
        
            if attempt == 3:
                label = "Yes"

        except Exception as e:
            # print error message
            print(e)
            output_text = "NULL Response"
            label = "Yes"
        return output_text, label

    def generate_text_factcheck(self, data) -> str:
        claim = "\nClaim: " + data["claim"]
        text_evidence = "\nText Evidence: " + data["evidence"]

        try:
            attempt = 0
            while attempt < 3:
                response = self.client.chat.completions.create(
                    model=self.MODEL,
                    messages=[
                        {"role": "system", "content": TEXT_FACTCHECK_PROMPT},
                        {"role": "user", "content": [
                            {"type": "text", "text": claim + text_evidence},
                        ]}
                    ],
                    temperature=0.0,
                )

                output_text = response.choices[0].message.content

                if check_label(output_text) is None:
                    attempt += 1
                    time.sleep(1)
                else:
                    label = check_label(output_text)
                    break

            if attempt == 3:
                label = "NEI"

        except Exception as e:
            # print error message
            print(e)
            output_text = "NULL Response"
            label = "NEI"
        return output_text, label

    def generate_multimodal_factcheck(self, data) -> str:
        img = self.encode_image(data['image'])
        claim = "\nClaim: " + data["claim"]
        image_evidence = "\nImage Evidence: "
        text_evidence = "\nText Evidence: " + data["evidence"]

        try:
            attempt = 0
            while attempt < 3:
                if isinstance(img, list):
                    print("Multiple image", len(img))
                    # build the “user” content as a list
                    user_content = []
                    # 1) initial text
                    user_content.append({
                        "type": "text",
                        "text": claim + image_evidence
                    })
                    # 2) one image_url entry for each base64 in img
                    for b64 in img:
                        user_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64}"
                            }
                        })
                    # 3) final text evidence
                    user_content.append({
                        "type": "text",
                        "text": text_evidence
                    })
                    response = self.client.chat.completions.create(
                        model=self.MODEL,
                        messages=[
                            {"role": "system", "content": MULTIMODAL_FACTCHECK_PROMPT},
                            {"role": "user", "content": user_content}
                        ],
                        temperature=0.0,
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.MODEL,
                        messages=[
                            {"role": "system", "content": MULTIMODAL_FACTCHECK_PROMPT},
                            {"role": "user", "content": [
                                {"type": "text", "text": claim + image_evidence},
                                {"type": "image_url", "image_url": {
                                    "url": f"data:image/png;base64,{img}"}
                                },
                                {"type": "text", "text": text_evidence}
                            ]}
                        ],
                        temperature=0.0,
                    )

                output_text = response.choices[0].message.content

                if check_label(output_text) is None:
                    attempt += 1
                    time.sleep(1)
                else:
                    label = check_label(output_text)
                    break

            if attempt == 3:
                label = "NEI"

        except Exception as e:
            # print error message
            print(e)
            output_text = "NULL Response"
            label = "NEI"
        return output_text, label

    def generate_claim_factcheck(self, data) -> str:
        claim = "\nClaim: " + data["claim"]

        try:
            attempt = 0
            while attempt < 3:
                response = self.client.chat.completions.create(
                    model=self.MODEL,
                    messages=[
                        {"role": "system", "content": CLAIM_FACTCHECK_PROMPT},
                        {"role": "user", "content": [
                            {"type": "text", "text": claim},
                        ]}
                    ],
                    temperature=0.0,
                )

                output_text = response.choices[0].message.content

                if check_label(output_text) is None:
                    attempt += 1
                    time.sleep(1)
                else:
                    label = check_label(output_text)
                    break

            if attempt == 3:
                label = "NEI"

        except Exception as e:
            # print error message
            print(e)
            output_text = "NULL Response"
            label = "NEI"
        return output_text, label

    def generate_analyze(self, data) -> str:
        img = self.encode_image(data['image'])
        claim = "\nClaim: " + data["claim"]
        image_evidence = "\nImage Evidence: "
        text_evidence = "\nText Evidence: " + data["evidence"]

        try:
            attempt = 0
            while attempt < 3:
                response = self.client.chat.completions.create(
                    model=self.MODEL,
                    messages=[
                        {"role": "system", "content": ANALYZE_PROMPT},
                        {"role": "user", "content": [
                            {"type": "text", "text": claim + image_evidence},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/png;base64,{img}"}
                            },
                            {"type": "text", "text": text_evidence}
                        ]}
                    ],
                    temperature=0.0,
                )

                output_text = response.choices[0].message.content

                if check_use_image(output_text) is None:
                    attempt += 1
                    time.sleep(1)
                else:
                    label = check_use_image(output_text)
                    break

            if attempt == 3:
                label = "NULL"

        except Exception as e:
            # print error message
            print(e)
            output_text = "NULL Response"
            label = "NULL"
        return output_text, label
    
    def generate_analyze_factcheck(self, data) -> str:
        img = self.encode_image(data['image'])
        claim = "\nClaim: " + data["claim"]
        image_evidence = "\nImage Evidence: "
        image_analysis = "\nImage Analysis: " + data["analyze"]
        text_evidence = "\nText Evidence: " + data["evidence"]

        try:
            attempt = 0
            while attempt < 3:
                response = self.client.chat.completions.create(
                    model=self.MODEL,
                    messages=[
                        {"role": "system", "content": ANALYZE_FACTCHECK_PROMPT},
                        {"role": "user", "content": [
                            {"type": "text", "text": claim + image_evidence},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/png;base64,{img}"}
                            },
                            {"type": "text", "text": image_analysis + text_evidence}
                        ]}
                    ],
                    temperature=0.0,
                )

                output_text = response.choices[0].message.content

                if check_label(output_text) is None:
                    attempt += 1
                    time.sleep(1)
                else:
                    label = check_label(output_text)
                    break

            if attempt == 3:
                label = "NEI"

        except Exception as e:
            # print error message
            print(e)
            output_text = "NULL Response"
            label = "NEI"
        return output_text, label
    
    def generate_justification(self, data) -> str:
        if data['image'] == "":
            return self.generate_text_justification(data)
        else:
            return self.generate_multimodal_justification(data)
    
    def generate_text_justification(self, data) -> str:
        claim = "\nClaim: " + data["claim"]
        text_evidence = "\nText Evidence: " + data["evidence"]
        verdict = "\nVerdict: " + data["label"]

        try:
            attempt = 0
            while attempt < 3:
                response = self.client.chat.completions.create(
                    model=self.MODEL,
                    messages=[
                        {"role": "system", "content": TEXT_JUSTIFICATION_PROMPT},
                        {"role": "user", "content": [
                            {"type": "text", "text": claim + text_evidence + verdict}
                        ]}
                    ],
                    temperature=0.0,
                )

                output_text = response.choices[0].message.content
                break

            if attempt == 3:
                output_text = "NULL Response"

        except Exception as e:
            # print error message
            print(e)
            output_text = "NULL Response"
        return output_text, data["label"]
    
    def generate_multimodal_justification(self, data) -> str:
        img = self.encode_image(data['image'])
        claim = "\nClaim: " + data["claim"]
        image_evidence = "\nImage Evidence: "
        text_evidence = "\nText Evidence: " + data["evidence"]
        verdict = "\nVerdict: " + data["label"]

        try:
            attempt = 0
            while attempt < 3:
                response = self.client.chat.completions.create(
                    model=self.MODEL,
                    messages=[
                        {"role": "system", "content": MULTIMODAL_JUSTIFICATION_PROMPT},
                        {"role": "user", "content": [
                            {"type": "text", "text": claim + image_evidence},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/png;base64,{img}"}
                            },
                            {"type": "text", "text": text_evidence + verdict}
                        ]}
                    ],
                    temperature=0.0,
                )

                output_text = response.choices[0].message.content
                break

            if attempt == 3:
                output_text = "NULL Response"

        except Exception as e:
            # print error message
            print(e)
            output_text = "NULL Response"
        return output_text, data["label"]


class gemini_model(openai_model):
    '''
    Base class for model compatibility implementation.
    '''
    def __init__(self, model_name) -> None:
        self.client = OpenAI(api_key=os.environ["GEMINI_API_KEY"], base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
        self.MODEL = model_name


class gemini_model_nonthink(openai_model):
    '''
    Gemini model implementation using native Google genai SDK with thinking budget disabled.
    '''
    def __init__(self, model_name) -> None:
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self.MODEL = model_name
    
    def generate_beforecheck(self, data) -> str:
        claim = "\nClaim: " + data["claim"]
        text_evidence = "\nText Evidence: " + data["evidence"]

        try:
            attempt = 0
            while attempt < 3:
                response = self.client.models.generate_content(
                    model=self.MODEL,
                    contents=BEFORECHECK_PROMPT + claim + text_evidence,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                        temperature=0.0,
                    ),
                )

                output_text = response.text

                if check_use_image(output_text) is None:
                    attempt += 1
                    time.sleep(1)
                else:
                    label = check_use_image(output_text)
                    break

            if attempt == 3:
                label = "Yes"

        except Exception as e:
            # print error message
            print(e)
            output_text = "NULL Response"
            label = "Yes"
        return output_text, label

    def generate_aftercheck(self, data) -> str:
        img = self.encode_image(data['image'])
        claim = "\nClaim: " + data["claim"]
        image_evidence = "\nImage Evidence: "
        text_evidence = "\nText Evidence: " + data["evidence"]

        try:
            attempt = 0
            while attempt < 3:
                response = self.client.models.generate_content(
                    model=self.MODEL,
                    contents=[
                        AFTERCHECK_PROMPT + claim + image_evidence,
                        types.Part.from_bytes(
                            data=base64.b64decode(img),
                            mime_type="image/png"
                        ),
                        text_evidence
                    ],
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                        temperature=0.0,
                    ),
                )

                output_text = response.text

                if check_use_image(output_text) is None:
                    attempt += 1
                    time.sleep(1)
                else:
                    label = check_use_image(output_text)
                    break
        
            if attempt == 3:
                label = "Yes"

        except Exception as e:
            # print error message
            print(e)
            output_text = "NULL Response"
            label = "Yes"
        return output_text, label

    def generate_text_factcheck(self, data) -> str:
        claim = "\nClaim: " + data["claim"]
        text_evidence = "\nText Evidence: " + data["evidence"]

        try:
            attempt = 0
            while attempt < 3:
                response = self.client.models.generate_content(
                    model=self.MODEL,
                    contents=TEXT_FACTCHECK_PROMPT + claim + text_evidence,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                        temperature=0.0,
                    ),
                )

                output_text = response.text

                if check_label(output_text) is None:
                    attempt += 1
                    time.sleep(1)
                else:
                    label = check_label(output_text)
                    break

            if attempt == 3:
                label = "NEI"

        except Exception as e:
            # print error message
            print(e)
            output_text = "NULL Response"
            label = "NEI"
        return output_text, label

    def generate_multimodal_factcheck(self, data) -> str:
        img = self.encode_image(data['image'])
        claim = "\nClaim: " + data["claim"]
        image_evidence = "\nImage Evidence: "
        text_evidence = "\nText Evidence: " + data["evidence"]

        try:
            attempt = 0
            while attempt < 3:
                if isinstance(img, list):
                    print("Multiple image", len(img))
                    # build the contents list
                    contents = [MULTIMODAL_FACTCHECK_PROMPT + claim + image_evidence]
                    # add each image
                    for b64 in img:
                        contents.append(types.Part.from_bytes(
                            data=base64.b64decode(b64),
                            mime_type="image/png"
                        ))
                    # add text evidence
                    contents.append(text_evidence)
                    
                    response = self.client.models.generate_content(
                        model=self.MODEL,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            thinking_config=types.ThinkingConfig(thinking_budget=0),
                            temperature=0.0,
                        ),
                    )
                else:
                    response = self.client.models.generate_content(
                        model=self.MODEL,
                        contents=[
                            MULTIMODAL_FACTCHECK_PROMPT + claim + image_evidence,
                            types.Part.from_bytes(
                                data=base64.b64decode(img),
                                mime_type="image/png"
                            ),
                            text_evidence
                        ],
                        config=types.GenerateContentConfig(
                            thinking_config=types.ThinkingConfig(thinking_budget=0),
                            temperature=0.0,
                        ),
                    )

                output_text = response.text

                if check_label(output_text) is None:
                    attempt += 1
                    time.sleep(1)
                else:
                    label = check_label(output_text)
                    break

            if attempt == 3:
                label = "NEI"

        except Exception as e:
            # print error message
            print(e)
            output_text = "NULL Response"
            label = "NEI"
        return output_text, label

    def generate_claim_factcheck(self, data) -> str:
        claim = "\nClaim: " + data["claim"]

        try:
            attempt = 0
            while attempt < 3:
                response = self.client.models.generate_content(
                    model=self.MODEL,
                    contents=CLAIM_FACTCHECK_PROMPT + claim,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                        temperature=0.0,
                    ),
                )

                output_text = response.text

                if check_label(output_text) is None:
                    attempt += 1
                    time.sleep(1)
                else:
                    label = check_label(output_text)
                    break

            if attempt == 3:
                label = "NEI"

        except Exception as e:
            # print error message
            print(e)
            output_text = "NULL Response"
            label = "NEI"
        return output_text, label

    def generate_analyze(self, data) -> str:
        img = self.encode_image(data['image'])
        claim = "\nClaim: " + data["claim"]
        image_evidence = "\nImage Evidence: "
        text_evidence = "\nText Evidence: " + data["evidence"]

        try:
            attempt = 0
            while attempt < 3:
                response = self.client.models.generate_content(
                    model=self.MODEL,
                    contents=[
                        ANALYZE_PROMPT + claim + image_evidence,
                        types.Part.from_bytes(
                            data=base64.b64decode(img),
                            mime_type="image/png"
                        ),
                        text_evidence
                    ],
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                        temperature=0.0,
                    ),
                )

                output_text = response.text

                if check_use_image(output_text) is None:
                    attempt += 1
                    time.sleep(1)
                else:
                    label = check_use_image(output_text)
                    break

            if attempt == 3:
                label = "NULL"

        except Exception as e:
            # print error message
            print(e)
            output_text = "NULL Response"
            label = "NULL"
        return output_text, label
    
    def generate_analyze_factcheck(self, data) -> str:
        img = self.encode_image(data['image'])
        claim = "\nClaim: " + data["claim"]
        image_evidence = "\nImage Evidence: "
        image_analysis = "\nImage Analysis: " + data["analyze"]
        text_evidence = "\nText Evidence: " + data["evidence"]

        try:
            attempt = 0
            while attempt < 3:
                response = self.client.models.generate_content(
                    model=self.MODEL,
                    contents=[
                        ANALYZE_FACTCHECK_PROMPT + claim + image_evidence,
                        types.Part.from_bytes(
                            data=base64.b64decode(img),
                            mime_type="image/png"
                        ),
                        image_analysis + text_evidence
                    ],
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                        temperature=0.0,
                    ),
                )

                output_text = response.text

                if check_label(output_text) is None:
                    attempt += 1
                    time.sleep(1)
                else:
                    label = check_label(output_text)
                    break

            if attempt == 3:
                label = "NEI"

        except Exception as e:
            # print error message
            print(e)
            output_text = "NULL Response"
            label = "NEI"
        return output_text, label


class gemini_model_think(openai_model):
    '''
    Gemini model implementation using native Google genai SDK with thinking budget enabled (128 tokens).
    '''
    def __init__(self, model_name) -> None:
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self.MODEL = model_name
    
    def generate_beforecheck(self, data) -> str:
        claim = "\nClaim: " + data["claim"]
        text_evidence = "\nText Evidence: " + data["evidence"]

        try:
            attempt = 0
            while attempt < 3:
                response = self.client.models.generate_content(
                    model=self.MODEL,
                    contents=BEFORECHECK_PROMPT + claim + text_evidence,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_budget=128),
                        temperature=0.0,
                    ),
                )

                output_text = response.text

                if check_use_image(output_text) is None:
                    attempt += 1
                    time.sleep(1)
                else:
                    label = check_use_image(output_text)
                    break

            if attempt == 3:
                label = "Yes"

        except Exception as e:
            # print error message
            print(e)
            output_text = "NULL Response"
            label = "Yes"
        return output_text, label

    def generate_aftercheck(self, data) -> str:
        img = self.encode_image(data['image'])
        claim = "\nClaim: " + data["claim"]
        image_evidence = "\nImage Evidence: "
        text_evidence = "\nText Evidence: " + data["evidence"]

        try:
            attempt = 0
            while attempt < 3:
                response = self.client.models.generate_content(
                    model=self.MODEL,
                    contents=[
                        AFTERCHECK_PROMPT + claim + image_evidence,
                        types.Part.from_bytes(
                            data=base64.b64decode(img),
                            mime_type="image/png"
                        ),
                        text_evidence
                    ],
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_budget=128),
                        temperature=0.0,
                    ),
                )

                output_text = response.text

                if check_use_image(output_text) is None:
                    attempt += 1
                    time.sleep(1)
                else:
                    label = check_use_image(output_text)
                    break
        
            if attempt == 3:
                label = "Yes"

        except Exception as e:
            # print error message
            print(e)
            output_text = "NULL Response"
            label = "Yes"
        return output_text, label

    def generate_text_factcheck(self, data) -> str:
        claim = "\nClaim: " + data["claim"]
        text_evidence = "\nText Evidence: " + data["evidence"]

        try:
            attempt = 0
            while attempt < 3:
                response = self.client.models.generate_content(
                    model=self.MODEL,
                    contents=TEXT_FACTCHECK_PROMPT + claim + text_evidence,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_budget=128),
                        temperature=0.0,
                    ),
                )

                output_text = response.text

                if check_label(output_text) is None:
                    attempt += 1
                    time.sleep(1)
                else:
                    label = check_label(output_text)
                    break

            if attempt == 3:
                label = "NEI"

        except Exception as e:
            # print error message
            print(e)
            output_text = "NULL Response"
            label = "NEI"
        return output_text, label

    def generate_multimodal_factcheck(self, data) -> str:
        img = self.encode_image(data['image'])
        claim = "\nClaim: " + data["claim"]
        image_evidence = "\nImage Evidence: "
        text_evidence = "\nText Evidence: " + data["evidence"]

        try:
            attempt = 0
            while attempt < 3:
                if isinstance(img, list):
                    print("Multiple image", len(img))
                    # build the contents list
                    contents = [MULTIMODAL_FACTCHECK_PROMPT + claim + image_evidence]
                    # add each image
                    for b64 in img:
                        contents.append(types.Part.from_bytes(
                            data=base64.b64decode(b64),
                            mime_type="image/png"
                        ))
                    # add text evidence
                    contents.append(text_evidence)
                    
                    response = self.client.models.generate_content(
                        model=self.MODEL,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            thinking_config=types.ThinkingConfig(thinking_budget=128),
                            temperature=0.0,
                        ),
                    )
                else:
                    response = self.client.models.generate_content(
                        model=self.MODEL,
                        contents=[
                            MULTIMODAL_FACTCHECK_PROMPT + claim + image_evidence,
                            types.Part.from_bytes(
                                data=base64.b64decode(img),
                                mime_type="image/png"
                            ),
                            text_evidence
                        ],
                        config=types.GenerateContentConfig(
                            thinking_config=types.ThinkingConfig(thinking_budget=128),
                            temperature=0.0,
                        ),
                    )

                output_text = response.text

                if check_label(output_text) is None:
                    attempt += 1
                    time.sleep(1)
                else:
                    label = check_label(output_text)
                    break

            if attempt == 3:
                label = "NEI"

        except Exception as e:
            # print error message
            print(e)
            output_text = "NULL Response"
            label = "NEI"
        return output_text, label

    def generate_claim_factcheck(self, data) -> str:
        claim = "\nClaim: " + data["claim"]

        try:
            attempt = 0
            while attempt < 3:
                response = self.client.models.generate_content(
                    model=self.MODEL,
                    contents=CLAIM_FACTCHECK_PROMPT + claim,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_budget=128),
                        temperature=0.0,
                    ),
                )

                output_text = response.text

                if check_label(output_text) is None:
                    attempt += 1
                    time.sleep(1)
                else:
                    label = check_label(output_text)
                    break

            if attempt == 3:
                label = "NEI"

        except Exception as e:
            # print error message
            print(e)
            output_text = "NULL Response"
            label = "NEI"
        return output_text, label

    def generate_analyze(self, data) -> str:
        img = self.encode_image(data['image'])
        claim = "\nClaim: " + data["claim"]
        image_evidence = "\nImage Evidence: "
        text_evidence = "\nText Evidence: " + data["evidence"]

        try:
            attempt = 0
            while attempt < 3:
                response = self.client.models.generate_content(
                    model=self.MODEL,
                    contents=[
                        ANALYZE_PROMPT + claim + image_evidence,
                        types.Part.from_bytes(
                            data=base64.b64decode(img),
                            mime_type="image/png"
                        ),
                        text_evidence
                    ],
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_budget=128),
                        temperature=0.0,
                    ),
                )

                output_text = response.text

                if check_use_image(output_text) is None:
                    attempt += 1
                    time.sleep(1)
                else:
                    label = check_use_image(output_text)
                    break

            if attempt == 3:
                label = "NULL"

        except Exception as e:
            # print error message
            print(e)
            output_text = "NULL Response"
            label = "NULL"
        return output_text, label
    
    def generate_analyze_factcheck(self, data) -> str:
        img = self.encode_image(data['image'])
        claim = "\nClaim: " + data["claim"]
        image_evidence = "\nImage Evidence: "
        image_analysis = "\nImage Analysis: " + data["analyze"]
        text_evidence = "\nText Evidence: " + data["evidence"]

        try:
            attempt = 0
            while attempt < 3:
                response = self.client.models.generate_content(
                    model=self.MODEL,
                    contents=[
                        ANALYZE_FACTCHECK_PROMPT + claim + image_evidence,
                        types.Part.from_bytes(
                            data=base64.b64decode(img),
                            mime_type="image/png"
                        ),
                        image_analysis + text_evidence
                    ],
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_budget=128),
                        temperature=0.0,
                    ),
                )

                output_text = response.text

                if check_label(output_text) is None:
                    attempt += 1
                    time.sleep(1)
                else:
                    label = check_label(output_text)
                    break

            if attempt == 3:
                label = "NEI"

        except Exception as e:
            # print error message
            print(e)
            output_text = "NULL Response"
            label = "NEI"
        return output_text, label


class claude_model(openai_model):
    '''
    Base class for model compatibility implementation.
    '''
    def __init__(self, model_name) -> None:
        self.client = OpenAI(api_key=os.environ["ANTHROPIC_API_KEY"], base_url="https://api.anthropic.com/v1/")
        self.MODEL = model_name

    def encode_image(self, image_path) -> str:
        if isinstance(image_path, str):
            img = Image.open(image_path)
            if img.width > 512 or img.height > 512:
                img.thumbnail((512, 512))

            if img.width < 256 or img.height < 256:
                new_w = max(img.width, 256)
                new_h = max(img.height, 256)
                img = img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)

            buffered = BytesIO()
            img.save(buffered, format="PNG")  # Always use PNG
            buffered.seek(0)

            return base64.b64encode(buffered.read()).decode("utf-8")
        else:
            image_list = []
            for path in image_path:
                img = Image.open(path)
                if img.width > 512 or img.height > 512:
                    img.thumbnail((512, 512))

                if img.width < 256 or img.height < 256:
                    new_w = max(img.width, 256)
                    new_h = max(img.height, 256)
                    img = img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)

                buffered = BytesIO()
                img.save(buffered, format="PNG")  # Always use PNG
                buffered.seek(0)
                image_list.append(base64.b64encode(buffered.read()).decode("utf-8"))
            return image_list


class qwenvl2_model():
    '''
    Base class for model compatibility implementation.
    '''
    def __init__(self, model_id) -> None:
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(
            model_id
        )

    def return_prompt(self, type) -> str:
        if type == "beforecheck":
            return BEFORECHECK_PROMPT
        elif type == "aftercheck":
            return AFTERCHECK_PROMPT
        elif type == "multimodal":
            return MULTIMODAL_FACTCHECK_PROMPT
        elif type == "text":
            return TEXT_FACTCHECK_PROMPT
        elif type == "claim":
            return CLAIM_FACTCHECK_PROMPT
        elif type == "analyze":
            return ANALYZE_PROMPT
        elif type == "analyze_factcheck":
            return ANALYZE_FACTCHECK_PROMPT
        else:
            raise ValueError("Invalid prompt type")

    def encode_image(self, image_path) -> str:
        if type(image_path) == str:
            # Open the image
            img = Image.open(image_path)

            # Resize the image if its width or height is greater than 512
            if img.width > 512 or img.height > 512:
                img.thumbnail((512, 512))

            if img.width < 256 or img.height < 256:
                new_w = max(img.width, 256)
                new_h = max(img.height, 256)
                img = img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)

            return img
        else:
            image_list = []
            for path in image_path:
                # Open the image
                img = Image.open(path)

                # Resize the image if its width or height is greater than 512
                if img.width > 512 or img.height > 512:
                    img.thumbnail((512, 512))

                if img.width < 256 or img.height < 256:
                    new_w = max(img.width, 256)
                    new_h = max(img.height, 256)
                    img = img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)

                image_list.append(img)
            return image_list

    def generate_check(self, inputs) -> str:
        try:
            attempt = 0
            while attempt < 3:
                output_ids = self.model.generate(**inputs, max_new_tokens=128, top_k=attempt+1)
                generated_ids = [
                    output_ids[len(input_ids) :]
                    for input_ids, output_ids in zip(inputs.input_ids, output_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                output_text = output_text[0]

                if check_use_image(output_text) is None:
                    attempt += 1
                else:
                    label = check_use_image(output_text)
                    break
            
            if attempt == 3:
                label = "Yes"

        except Exception as e:
            # print error message
            print(e)
            output_text = "NULL Response"
            label = "Yes"
        return output_text, label

    def generate_verification(self, inputs) -> str:
        try:
            attempt = 0
            while attempt < 3:
                output_ids = self.model.generate(**inputs, max_new_tokens=128, top_k=attempt+1)
                generated_ids = [
                    output_ids[len(input_ids) :]
                    for input_ids, output_ids in zip(inputs.input_ids, output_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                output_text = output_text[0]

                if check_label(output_text) is None:
                    attempt += 1
                else:
                    label = check_label(output_text)
                    break
            
            if attempt == 3:
                label = "NEI"

        except Exception as e:
            # print error message
            print(e)
            output_text = "NULL Response"
            label = "NEI"
        return output_text, label

    def generate_beforecheck(self, data) -> str:
        claim = "\nClaim: " + data["claim"]
        text_evidence = "\nText Evidence: " + data["evidence"]

        cont = []
        cont.append({'type': 'text', 'text': BEFORECHECK_PROMPT+claim+text_evidence})

        messages = [
            {
                "role": "user",
                "content": cont,
            }
        ]

        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.processor(
            text=[text_prompt], padding=True, return_tensors="pt"
        )
        inputs = inputs.to("cuda")

        return self.generate_check(inputs)

    def generate_aftercheck(self, data) -> str:
        img = self.encode_image(data['image'])
        claim = "\nClaim: " + data["claim"]
        image_evidence = "\nImage Evidence: "
        text_evidence = "\nText Evidence: " + data["evidence"]

        cont = []
        cont.append({'type': 'text' , 'text': AFTERCHECK_PROMPT+claim+image_evidence})
        cont.append({'type': 'image', 'image': img})
        cont.append({'type': 'text', 'text': text_evidence})

        messages = [
            {
                "role": "user",
                "content": cont,
            }
        ]

        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.processor(
            text=[text_prompt], images=[img], padding=True, return_tensors="pt"
        )
        inputs = inputs.to("cuda")

        return self.generate_check(inputs)


    def generate_multimodal_factcheck(self, data) -> str:
        img = self.encode_image(data['image'])
        claim = "\nClaim: " + data["claim"]
        image_evidence = "\nImage Evidence: "
        text_evidence = "\nText Evidence: " + data["evidence"]

        cont = []
        cont.append({'type': 'text' , 'text': MULTIMODAL_FACTCHECK_PROMPT+claim+image_evidence})

        if isinstance(img, list):
            for b64 in img:
                cont.append({'type': 'image', 'image': b64})
        else:
            cont.append({'type': 'image', 'image': img})

        cont.append({'type': 'text', 'text': text_evidence})

        messages = [
            {
                "role": "user",
                "content": cont,
            }
        ]

        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)


        if isinstance(img, list):
            print("multiple image", len(img))
            inputs = self.processor(
                text=[text_prompt], images=img, padding=True, return_tensors="pt"
            )
        else:
            inputs = self.processor(
                text=[text_prompt], images=[img], padding=True, return_tensors="pt"
            )
        inputs = inputs.to("cuda")

        return self.generate_verification(inputs)

    def generate_text_factcheck(self, data) -> str:
        claim = "\nClaim: " + data["claim"]
        text_evidence = "\nText Evidence: " + data["evidence"]

        cont = []
        cont.append({'type': 'text', 'text': TEXT_FACTCHECK_PROMPT+claim+text_evidence})

        messages = [
            {
                "role": "user",
                "content": cont,
            }
        ]

        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.processor(
            text=[text_prompt], padding=True, return_tensors="pt"
        )
        inputs = inputs.to("cuda")

        return self.generate_verification(inputs)

    def generate_claim_factcheck(self, data) -> str:
        claim = "\nClaim: " + data["claim"]

        cont = []
        cont.append({'type': 'text', 'text': CLAIM_FACTCHECK_PROMPT+claim})

        messages = [
            {
                "role": "user",
                "content": cont,
            }
        ]

        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.processor(
            text=[text_prompt], padding=True, return_tensors="pt"
        )
        inputs = inputs.to("cuda")

        return self.generate_verification(inputs)
    
    def generate_analyze(self, data) -> str:
        img = self.encode_image(data['image'])
        claim = "\nClaim: " + data["claim"]
        image_evidence = "\nImage Evidence: "
        text_evidence = "\nText Evidence: " + data["evidence"]

        cont = []
        cont.append({'type': 'text' , 'text': ANALYZE_PROMPT+claim+image_evidence})
        cont.append({'type': 'image', 'image': img})
        cont.append({'type': 'text', 'text': text_evidence})

        messages = [
            {
                "role": "user",
                "content": cont,
            }
        ]

        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.processor(
            text=[text_prompt], images=[img], padding=True, return_tensors="pt"
        )
        inputs = inputs.to("cuda")

        return self.generate_check(inputs)

    def generate_analyze_factcheck(self, data) -> str:
        img = self.encode_image(data['image'])
        claim = "\nClaim: " + data["claim"]
        image_evidence = "\nImage Evidence: "
        image_analysis = "\nImage Analysis: " + data["analyze"]
        text_evidence = "\nText Evidence: " + data["evidence"]

        cont = []
        cont.append({'type': 'text' , 'text': ANALYZE_FACTCHECK_PROMPT+claim+image_evidence})
        cont.append({'type': 'image', 'image': img})
        cont.append({'type': 'text', 'text': image_analysis + text_evidence})

        messages = [
            {
                "role": "user",
                "content": cont,
            }
        ]

        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.processor(
            text=[text_prompt], images=[img], padding=True, return_tensors="pt"
        )
        inputs = inputs.to("cuda")

        return self.generate_verification(inputs)


class qwen2_5_vl_model():
    '''
    Base class for Qwen2.5-VL model compatibility implementation.
    '''
    def __init__(self, model_id) -> None:
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(
            model_id
        )

    def return_prompt(self, type) -> str:
        if type == "beforecheck":
            return BEFORECHECK_PROMPT
        elif type == "aftercheck":
            return AFTERCHECK_PROMPT
        elif type == "multimodal":
            return MULTIMODAL_FACTCHECK_PROMPT
        elif type == "text":
            return TEXT_FACTCHECK_PROMPT
        elif type == "claim":
            return CLAIM_FACTCHECK_PROMPT
        elif type == "analyze":
            return ANALYZE_PROMPT
        elif type == "analyze_factcheck":
            return ANALYZE_FACTCHECK_PROMPT
        else:
            raise ValueError("Invalid prompt type")

    def encode_image(self, image_path) -> str:
        if type(image_path) == str:
            # Open the image
            img = Image.open(image_path)

            # Resize the image if its width or height is greater than 512
            if img.width > 512 or img.height > 512:
                img.thumbnail((512, 512))

            if img.width < 256 or img.height < 256:
                new_w = max(img.width, 256)
                new_h = max(img.height, 256)
                img = img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)

            return img
        else:
            image_list = []
            for path in image_path:
                # Open the image
                img = Image.open(path)

                # Resize the image if its width or height is greater than 512
                if img.width > 512 or img.height > 512:
                    img.thumbnail((512, 512))

                if img.width < 256 or img.height < 256:
                    new_w = max(img.width, 256)
                    new_h = max(img.height, 256)
                    img = img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)

                image_list.append(img)
            return image_list

    def generate_check(self, inputs) -> str:
        try:
            attempt = 0
            while attempt < 3:
                output_ids = self.model.generate(**inputs, max_new_tokens=128, top_k=attempt+1)
                generated_ids = [
                    output_ids[len(input_ids) :]
                    for input_ids, output_ids in zip(inputs.input_ids, output_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                output_text = output_text[0]

                if check_use_image(output_text) is None:
                    attempt += 1
                else:
                    label = check_use_image(output_text)
                    break

            if attempt == 3:
                label = "Yes"

        except Exception as e:
            # print error message
            print(e)
            output_text = "NULL Response"
            label = "Yes"
        return output_text, label

    def generate_verification(self, inputs) -> str:
        try:
            attempt = 0
            while attempt < 3:
                output_ids = self.model.generate(**inputs, max_new_tokens=128, top_k=attempt+1)
                generated_ids = [
                    output_ids[len(input_ids) :]
                    for input_ids, output_ids in zip(inputs.input_ids, output_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                output_text = output_text[0]

                if check_label(output_text) is None:
                    attempt += 1
                else:
                    label = check_label(output_text)
                    break

            if attempt == 3:
                label = "NEI"

        except Exception as e:
            # print error message
            print(e)
            output_text = "NULL Response"
            label = "NEI"
        return output_text, label

    def generate_beforecheck(self, data) -> str:
        claim = "\nClaim: " + data["claim"]
        text_evidence = "\nText Evidence: " + data["evidence"]

        cont = []
        cont.append({'type': 'text', 'text': BEFORECHECK_PROMPT+claim+text_evidence})

        messages = [
            {
                "role": "user",
                "content": cont,
            }
        ]

        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.processor(
            text=[text_prompt], padding=True, return_tensors="pt"
        )
        inputs = inputs.to("cuda")

        return self.generate_check(inputs)

    def generate_aftercheck(self, data) -> str:
        img = self.encode_image(data['image'])
        claim = "\nClaim: " + data["claim"]
        image_evidence = "\nImage Evidence: "
        text_evidence = "\nText Evidence: " + data["evidence"]

        cont = []
        cont.append({'type': 'text' , 'text': AFTERCHECK_PROMPT+claim+image_evidence})
        cont.append({'type': 'image', 'image': img})
        cont.append({'type': 'text', 'text': text_evidence})

        messages = [
            {
                "role": "user",
                "content": cont,
            }
        ]

        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.processor(
            text=[text_prompt], images=[img], padding=True, return_tensors="pt"
        )
        inputs = inputs.to("cuda")

        return self.generate_check(inputs)

    def generate_multimodal_factcheck(self, data) -> str:
        img = self.encode_image(data['image'])
        claim = "\nClaim: " + data["claim"]
        image_evidence = "\nImage Evidence: "
        text_evidence = "\nText Evidence: " + data["evidence"]

        cont = []
        cont.append({'type': 'text' , 'text': MULTIMODAL_FACTCHECK_PROMPT+claim+image_evidence})

        if isinstance(img, list):
            for b64 in img:
                cont.append({'type': 'image', 'image': b64})
        else:
            cont.append({'type': 'image', 'image': img})

        cont.append({'type': 'text', 'text': text_evidence})

        messages = [
            {
                "role": "user",
                "content": cont,
            }
        ]

        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        if isinstance(img, list):
            print("multiple image", len(img))
            inputs = self.processor(
                text=[text_prompt], images=img, padding=True, return_tensors="pt"
            )
        else:
            inputs = self.processor(
                text=[text_prompt], images=[img], padding=True, return_tensors="pt"
            )
        inputs = inputs.to("cuda")

        return self.generate_verification(inputs)

    def generate_text_factcheck(self, data) -> str:
        claim = "\nClaim: " + data["claim"]
        text_evidence = "\nText Evidence: " + data["evidence"]

        cont = []
        cont.append({'type': 'text', 'text': TEXT_FACTCHECK_PROMPT+claim+text_evidence})

        messages = [
            {
                "role": "user",
                "content": cont,
            }
        ]

        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.processor(
            text=[text_prompt], padding=True, return_tensors="pt"
        )
        inputs = inputs.to("cuda")

        return self.generate_verification(inputs)

    def generate_claim_factcheck(self, data) -> str:
        claim = "\nClaim: " + data["claim"]

        cont = []
        cont.append({'type': 'text', 'text': CLAIM_FACTCHECK_PROMPT+claim})

        messages = [
            {
                "role": "user",
                "content": cont,
            }
        ]

        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.processor(
            text=[text_prompt], padding=True, return_tensors="pt"
        )
        inputs = inputs.to("cuda")

        return self.generate_verification(inputs)

    def generate_analyze(self, data) -> str:
        img = self.encode_image(data['image'])
        claim = "\nClaim: " + data["claim"]
        image_evidence = "\nImage Evidence: "
        text_evidence = "\nText Evidence: " + data["evidence"]

        cont = []
        cont.append({'type': 'text' , 'text': ANALYZE_PROMPT+claim+image_evidence})
        cont.append({'type': 'image', 'image': img})
        cont.append({'type': 'text', 'text': text_evidence})

        messages = [
            {
                "role": "user",
                "content": cont,
            }
        ]

        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.processor(
            text=[text_prompt], images=[img], padding=True, return_tensors="pt"
        )
        inputs = inputs.to("cuda")

        return self.generate_check(inputs)

    def generate_analyze_factcheck(self, data) -> str:
        img = self.encode_image(data['image'])
        claim = "\nClaim: " + data["claim"]
        image_evidence = "\nImage Evidence: "
        image_analysis = "\nImage Analysis: " + data["analyze"]
        text_evidence = "\nText Evidence: " + data["evidence"]

        cont = []
        cont.append({'type': 'text' , 'text': ANALYZE_FACTCHECK_PROMPT+claim+image_evidence})
        cont.append({'type': 'image', 'image': img})
        cont.append({'type': 'text', 'text': image_analysis + text_evidence})

        messages = [
            {
                "role": "user",
                "content": cont,
            }
        ]

        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.processor(
            text=[text_prompt], images=[img], padding=True, return_tensors="pt"
        )
        inputs = inputs.to("cuda")

        return self.generate_verification(inputs)


class llama3_vision_model():
    '''
    Base class for model compatibility implementation.
    '''
    def __init__(self, model_id) -> None:
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(
            model_id
        )
        self.llamaprompt = "Do not add any justification."

    def return_prompt(self, type) -> str:
        if type == "beforecheck":
            return BEFORECHECK_PROMPT+self.llamaprompt
        elif type == "aftercheck":
            return AFTERCHECK_PROMPT+self.llamaprompt
        elif type == "multimodal":
            return MULTIMODAL_FACTCHECK_PROMPT+self.llamaprompt
        elif type == "text":
            return TEXT_FACTCHECK_PROMPT+self.llamaprompt
        elif type == "claim":
            return CLAIM_FACTCHECK_PROMPT+self.llamaprompt
        elif type == "analyze":
            return ANALYZE_PROMPT
        elif type == "analyze_factcheck":
            return ANALYZE_FACTCHECK_PROMPT+self.llamaprompt
        else:
            raise ValueError("Invalid prompt type")

    def encode_image(self, image_path) -> str:
        if type(image_path) == str:
            # Open the image
            img = Image.open(image_path)

            # Resize the image if its width or height is greater than 512
            if img.width > 512 or img.height > 512:
                img.thumbnail((512, 512))

            if img.width < 256 or img.height < 256:
                new_w = max(img.width, 256)
                new_h = max(img.height, 256)
                img = img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)

            return img
        else:
            image_list = []
            for path in image_path:
                # Open the image
                img = Image.open(path)

                # Resize the image if its width or height is greater than 512
                if img.width > 512 or img.height > 512:
                    img.thumbnail((512, 512))

                if img.width < 256 or img.height < 256:
                    new_w = max(img.width, 256)
                    new_h = max(img.height, 256)
                    img = img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)

                image_list.append(img)
            return image_list

    def generate_check(self, inputs) -> str:
        try:
            attempt = 0
            while attempt < 3:
                output_ids = self.model.generate(**inputs, max_new_tokens=100, top_k=attempt+1)
                generated_ids = [
                    output_ids[len(input_ids) :]
                    for input_ids, output_ids in zip(inputs.input_ids, output_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                output_text = output_text[0]

                if check_use_image(output_text) is None:
                    attempt += 1
                else:
                    label = check_use_image(output_text)
                    break
            
            if attempt == 3:
                label = "Yes"

        except Exception as e:
            # print error message
            print(e)
            output_text = "NULL Response"
            label = "Yes"
        return output_text, label

    def generate_verification(self, inputs) -> str:
        try:
            attempt = 0
            while attempt < 3:
                output_ids = self.model.generate(**inputs, max_new_tokens=100, top_k=attempt+1)
                generated_ids = [
                    output_ids[len(input_ids) :]
                    for input_ids, output_ids in zip(inputs.input_ids, output_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                output_text = output_text[0]

                if check_label(output_text) is None:
                    attempt += 1
                else:
                    label = check_label(output_text)
                    break
            
            if attempt == 3:
                label = "NEI"

        except Exception as e:
            # print error message
            print(e)
            output_text = "NULL Response"
            label = "NEI"
        return output_text, label

    def generate_beforecheck(self, data) -> str:
        claim = "\nClaim: " + data["claim"]
        text_evidence = "\nText Evidence: " + data["evidence"]

        cont = []
        cont.append({'type': 'text', 'text': BEFORECHECK_PROMPT+self.llamaprompt+claim+text_evidence})

        messages = [
            {
                "role": "user",
                "content": cont,
            }
        ]

        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.processor(
            text=[text_prompt], padding=True, return_tensors="pt"
        )
        inputs = inputs.to("cuda")

        return self.generate_check(inputs)

    def generate_aftercheck(self, data) -> str:
        img = self.encode_image(data['image'])
        claim = "\nClaim: " + data["claim"]
        text_evidence = "\nText Evidence: " + data["evidence"]

        cont = []
        cont.append({'type': 'image'})
        cont.append({'type': 'text' , 'text': AFTERCHECK_PROMPT+self.llamaprompt+claim})
        cont.append({'type': 'text', 'text': text_evidence})

        messages = [
            {
                "role": "user",
                "content": cont,
            }
        ]

        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.processor(
            text=[text_prompt], images=[img], padding=True, return_tensors="pt"
        )
        inputs = inputs.to("cuda")

        return self.generate_check(inputs)


    def generate_multimodal_factcheck(self, data) -> str:
        img = self.encode_image(data['image'])
        claim = "\nClaim: " + data["claim"]
        text_evidence = "\nText Evidence: " + data["evidence"]

        cont = []
        cont.append({'type': 'image'})
        cont.append({'type': 'text' , 'text': MULTIMODAL_FACTCHECK_PROMPT+self.llamaprompt+claim})
        cont.append({'type': 'text', 'text': text_evidence})

        messages = [
            {
                "role": "user",
                "content": cont,
            }
        ]

        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)


        if isinstance(img, list):
            print("multiple image", len(img))
            inputs = self.processor(
                text=[text_prompt], images=img, padding=True, return_tensors="pt"
            )
        else:
            inputs = self.processor(
                text=[text_prompt], images=[img], padding=True, return_tensors="pt"
            )
        inputs = inputs.to("cuda")

        return self.generate_verification(inputs)

    def generate_text_factcheck(self, data) -> str:
        claim = "\nClaim: " + data["claim"]
        text_evidence = "\nText Evidence: " + data["evidence"]

        cont = []
        cont.append({'type': 'text', 'text': TEXT_FACTCHECK_PROMPT+self.llamaprompt+claim+text_evidence})

        messages = [
            {
                "role": "user",
                "content": cont,
            }
        ]

        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.processor(
            text=[text_prompt], padding=True, return_tensors="pt"
        )
        inputs = inputs.to("cuda")

        return self.generate_verification(inputs)

    def generate_claim_factcheck(self, data) -> str:
        claim = "\nClaim: " + data["claim"]

        cont = []
        cont.append({'type': 'text', 'text': CLAIM_FACTCHECK_PROMPT+self.llamaprompt+claim})

        messages = [
            {
                "role": "user",
                "content": cont,
            }
        ]

        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.processor(
            text=[text_prompt], padding=True, return_tensors="pt"
        )
        inputs = inputs.to("cuda")

        return self.generate_verification(inputs)
    
    def generate_analyze(self, data) -> str:
        img = self.encode_image(data['image'])
        claim = "\nClaim: " + data["claim"]
        text_evidence = "\nText Evidence: " + data["evidence"]

        cont = []
        cont.append({'type': 'image', 'image': img})
        cont.append({'type': 'text' , 'text': ANALYZE_PROMPT + claim + text_evidence})

        messages = [
            {
                "role": "user",
                "content": cont,
            }
        ]

        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.processor(
            text=[text_prompt], images=[img], padding=True, return_tensors="pt"
        )
        inputs = inputs.to("cuda")

        return self.generate_check(inputs)

    def generate_analyze_factcheck(self, data) -> str:
        img = self.encode_image(data['image'])
        claim = "\nClaim: " + data["claim"]
        image_analysis = "\nImage Analysis: " + data["analyze"]
        text_evidence = "\nText Evidence: " + data["evidence"]

        cont = []
        cont.append({'type': 'image', 'image': img})
        cont.append({'type': 'text' , 'text': ANALYZE_FACTCHECK_PROMPT + self.llamaprompt + claim + image_analysis + text_evidence})

        messages = [
            {
                "role": "user",
                "content": cont,
            }
        ]

        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.processor(
            text=[text_prompt], images=[img], padding=True, return_tensors="pt"
        )
        inputs = inputs.to("cuda")

        return self.generate_verification(inputs)


# InternVL3 helper functions
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image_internvl3(image_file, input_size=448, max_num=12):
    img = Image.open(image_file).convert('RGB')
    if img.width > 512 or img.height > 512:
        img.thumbnail((512, 512))

    if img.width < 256 or img.height < 256:
        new_w = max(img.width, 256)
        new_h = max(img.height, 256)
        img = img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

class internvl3_model():
    '''
    Base class for model compatibility implementation.
    '''
    def __init__(self, model_id) -> None:
        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map="auto"
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
        self.generation_config = dict(max_new_tokens=1024, do_sample=True)

    def return_prompt(self, type) -> str:
        if type == "beforecheck":
            return BEFORECHECK_PROMPT
        elif type == "aftercheck":
            return AFTERCHECK_PROMPT
        elif type == "multimodal":
            return MULTIMODAL_FACTCHECK_PROMPT
        elif type == "text":
            return TEXT_FACTCHECK_PROMPT
        elif type == "claim":
            return CLAIM_FACTCHECK_PROMPT
        elif type == "analyze":
            return ANALYZE_PROMPT
        elif type == "analyze_factcheck":
            return ANALYZE_FACTCHECK_PROMPT
        else:
            raise ValueError("Invalid prompt type")

    def encode_image(self, image_path) -> str:
        if type(image_path) == str:
            # Load and process a single image
            pixel_values = load_image_internvl3(image_path, max_num=12).to(torch.bfloat16).cuda()
            return pixel_values
        else:
            # Load and process multiple images
            pixel_values_list = []
            for path in image_path:
                pixel_values = load_image_internvl3(path, max_num=12).to(torch.bfloat16).cuda()
                pixel_values_list.append(pixel_values)
            return pixel_values_list

    def generate_check(self, question, pixel_values=None, history=None, num_patches_list=None) -> str:
        try:
            attempt = 0
            while attempt < 3:
                response, _ = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    question,
                    self.generation_config,
                    history=history,
                    num_patches_list=num_patches_list,
                    return_history=True
                )

                if check_use_image(response) is None:
                    attempt += 1
                else:
                    label = check_use_image(response)
                    break

            if attempt == 3:
                label = "Yes"

        except Exception as e:
            # print error message
            print(e)
            response = "NULL Response"
            label = "Yes"
        return response, label

    def generate_verification(self, question, pixel_values=None, history=None, num_patches_list=None) -> str:
        try:
            attempt = 0
            while attempt < 3:
                response, _ = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    question,
                    self.generation_config,
                    history=history,
                    num_patches_list=num_patches_list,
                    return_history=True
                )

                if check_label(response) is None:
                    attempt += 1
                else:
                    label = check_label(response)
                    break

            if attempt == 3:
                label = "NEI"

        except Exception as e:
            # print error message
            print(e)
            response = "NULL Response"
            label = "NEI"
        return response, label

    def generate_beforecheck(self, data) -> str:
        claim = "\nClaim: " + data["claim"]
        text_evidence = "\nText Evidence: " + data["evidence"]
        question = BEFORECHECK_PROMPT + claim + text_evidence

        return self.generate_check(question)

    def generate_aftercheck(self, data) -> str:
        pixel_values = self.encode_image(data['image'])
        claim = "\nClaim: " + data["claim"]
        image_evidence = "\nImage Evidence: "
        text_evidence = "\nText Evidence: " + data["evidence"]
        question = AFTERCHECK_PROMPT + claim + image_evidence + "<image>" + text_evidence

        return self.generate_check(question, pixel_values)

    def generate_multimodal_factcheck(self, data) -> str:
        pixel_values = self.encode_image(data['image'])
        claim = "\nClaim: " + data["claim"]
        image_evidence = "\nImage Evidence: "
        text_evidence = "\nText Evidence: " + data["evidence"]

        if isinstance(pixel_values, list):
            # Multiple images
            print("Multiple image", len(pixel_values))
            # Concatenate all pixel values
            combined_pixel_values = torch.cat(pixel_values, dim=0)
            num_patches_list = [pv.size(0) for pv in pixel_values]
            question = MULTIMODAL_FACTCHECK_PROMPT + claim + image_evidence
            # Add image placeholders for each image
            question += "<image>" * len(pixel_values)
            question += text_evidence
            return self.generate_verification(question, combined_pixel_values, num_patches_list=num_patches_list)
        else:
            # Single image
            question = MULTIMODAL_FACTCHECK_PROMPT + claim + image_evidence + "<image>" + text_evidence
            return self.generate_verification(question, pixel_values)

    def generate_text_factcheck(self, data) -> str:
        claim = "\nClaim: " + data["claim"]
        text_evidence = "\nText Evidence: " + data["evidence"]
        question = TEXT_FACTCHECK_PROMPT + claim + text_evidence

        return self.generate_verification(question)

    def generate_claim_factcheck(self, data) -> str:
        claim = "\nClaim: " + data["claim"]
        question = CLAIM_FACTCHECK_PROMPT + claim

        return self.generate_verification(question)
