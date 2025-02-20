import os
import json
import torch
import argparse
import jinja2
from tqdm import tqdm
from openai import OpenAI
from diffusers import DiffusionPipeline, StableDiffusion3Pipeline

from utils.utils import build_metadata


env = jinja2.Environment(loader=jinja2.FileSystemLoader('prompts'))
template = env.get_template('prompt_data_gen.jinja')
OPENAI_API_KEY = "api-key"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
client = OpenAI()

# we use gpt-4o-mini as default: current pricing is 
# $0.150 / 1M input tokens, $0.600 / 1M output tokens
def generate_gpt_response(prompt,messages=[], model="gpt-4o-mini", temperature=0.7, json=False):
    messages.append({"role": "user", "content": prompt})
      
    if json:      
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature = temperature,
            response_format={ "type": "json_object" }
        )
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature = temperature,
        )
    content=response.choices[0].message.content
    messages.append({'role': 'assistant', 'content': content})
    return messages


def generate_target_answer_with_image_caption(question, correct_answer):
    prompt = template.render(
        question=question,
        correct_answer=correct_answer,
    )
    messages = generate_gpt_response(prompt, [], json=True)
    response = messages[-1]['content']
    json_response = json.loads(response)
    
    return json_response['wrong_answer'], json_response['poison_image_caption']


def gen_poison_imgs(args, captions, qids):
    print("===Pipeline Loading===")
    if args.t2i_model_type == 'stable':
        pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16)
        pipe = pipe.to("cuda")
    elif args.t2i_model_type == 'latent':
        pipe = DiffusionPipeline.from_pretrained("CompVis/ldm-text2im-large-256").to("cuda")
    else:
        raise NotImplementedError
    print("===Pipeline Loading Complete===")

    poisoned_img_paths = [] 
    for caption, qid in tqdm(zip(captions, qids), desc='Generating images'):
        with torch.no_grad():  # Turn off gradients to save memory
            images = pipe(
                [caption],
                num_inference_steps=28,
                guidance_scale=3.5,
            ).images
        save_path = f"{args.save_img_dir}/{qid}.png"
        images[0].save(save_path)
        poisoned_img_paths.append(save_path)
    return poisoned_img_paths
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="MMQA", help=["MMQA", "WebQA"])
    parser.add_argument("--metadata_path", type=str)
    parser.add_argument("--t2i_model_type", type=str, default='stable', help=['stable', 'latent'])
    parser.add_argument("--save_data_dir", type=str, default='./results', help="save dir path for metadata")
    parser.add_argument("--save_img_dir", type=str, default='./results', help="save dir path for poisoned images")
    args = parser.parse_args()
    
    os.makedirs(args.save_data_dir, exist_ok=True)
    os.makedirs(args.save_img_dir, exist_ok=True)

    with open(args.metadata_path, 'r') as f:
        qa_data = json.load(f)

    target_answers, poison_img_captions, qids = [], [], []
    gt_answers, questions = [], []
       
    cnt = 0
    for item in tqdm(qa_data):
        if args.task == 'MMQA':
            qid = item["qid"]
            question = item["question"]
            original_answer = item["answers"][0]["answer"]
        else:
            qid = item
            original_sample = qa_data[item]
            question = original_sample['Q'][1:-1]
            original_answer = original_sample["A"][0][1:-1]

        max_try = 3
        while max_try > 0:
            try:
                wrong_answer, poison_image_caption = generate_target_answer_with_image_caption(question, original_answer)
                target_answers.append(wrong_answer)
                poison_img_captions.append(poison_image_caption)
                gt_answers.append(original_answer)
                questions.append(question)
                qids.append(qid)
                cnt += 1
            except Exception as e:
                print(e)
                max_try -= 1
        
        if cnt == 5:
            break

    poisoned_img_paths = gen_poison_imgs(args, poison_img_captions, qids)
    build_metadata(
        args.task, args.metadata_path, args.save_data_dir, poisoned_img_paths, 
        target_answers=target_answers, gt_answers=gt_answers, questions=questions, poison_img_captions=poison_img_captions, poison_type='lpa-bb'
    )
