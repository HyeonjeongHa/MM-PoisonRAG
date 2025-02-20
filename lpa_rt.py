import os
from PIL import Image
import torch
import argparse
from torch import nn
from torchvision import transforms
import json
import torch.nn.functional as F
from transformers import (
    AutoProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)
from tqdm import tqdm
from utils.utils import build_metadata


clip_vision_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14-336").to("cuda")
clip_vision_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14-336")
clip_text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14-336").to("cuda")


def generate_adversarial_image(
    image_path,
    query,
    ouput_image_path,
    clip_vision_model,
    clip_text_model,
    clip_tokenizer,
    num_steps=10,
    epsilon=0.01,
    alpha=0.005,
):
    # Load the image
    image = Image.open(image_path).convert("RGB")

    # Define the preprocessing transforms
    preprocess = transforms.Compose(
        [
            transforms.Resize(336),
            transforms.CenterCrop(336),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=clip_vision_processor.image_processor.image_mean,
                std=clip_vision_processor.image_processor.image_std,
            ),
        ]
    )

    # Preprocess the image
    image_tensor = preprocess(image).unsqueeze(0).to("cuda")  # Add batch dimension
    image_tensor.requires_grad = True  # Enable gradient computation

    # Get the text embedding
    text_inputs = clip_tokenizer([query], return_tensors="pt").to("cuda")
    text_outputs = clip_text_model(**text_inputs)
    text_embeds = text_outputs.text_embeds  # shape: [batch_size, hidden_size]
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    # Detach text_embeds from the computational graph
    text_embeds = text_embeds.detach()

    # Ensure text_embeds does not require gradients
    text_embeds.requires_grad = False

    # Optimization loop
    optimizer = torch.optim.Adam([image_tensor], lr=alpha)
    loss_fn = nn.CosineSimilarity(dim=-1)

    original_image_tensor = image_tensor.clone().detach()

    # show a progress bar with loss
    progress_bar = tqdm(range(num_steps), desc="Adversarial Attack")
    for step in range(num_steps):
        optimizer.zero_grad()

        # Get image embedding
        outputs = clip_vision_model(pixel_values=image_tensor)
        image_embeds = outputs.image_embeds  # shape: [batch_size, hidden_size]
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        # Compute cosine similarity
        similarity = loss_fn(image_embeds, text_embeds)
        loss = -similarity.mean()  # Negative to maximize similarity
        progress_bar.set_postfix({"Loss": loss.item()})
        loss.backward()
        optimizer.step()

        # Clip the perturbation to be within epsilon
        perturbation = torch.clamp(
            image_tensor - original_image_tensor, -epsilon, epsilon
        )
        image_tensor.data = original_image_tensor + perturbation

        # Clamp image_tensor to valid range after de-normalization
        mean = torch.tensor(clip_vision_processor.image_processor.image_mean).view(
            1, -1, 1, 1
        ).to("cuda")
        std = torch.tensor(clip_vision_processor.image_processor.image_std).view(
            1, -1, 1, 1
        ).to("cuda")
        unnormalized_image = image_tensor * std + mean
        unnormalized_image = torch.clamp(unnormalized_image, 0, 1)
        image_tensor.data = (unnormalized_image - mean) / std
        progress_bar.update(1)

    # After optimization, save the adversarial image
    # Reverse the preprocessing to get the image back
    adversarial_image = image_tensor.detach()

    # De-normalize
    mean = torch.tensor(clip_vision_processor.image_processor.image_mean).view(
        1, -1, 1, 1
    ).to("cuda")
    std = torch.tensor(clip_vision_processor.image_processor.image_std).view(
        1, -1, 1, 1
    ).to("cuda")
    adversarial_image = adversarial_image * std + mean

    # Clamp to [0,1]
    adversarial_image = torch.clamp(adversarial_image, 0, 1)
    adversarial_image = adversarial_image.squeeze(0)  # Remove batch dimension

    # Convert to numpy array
    adversarial_image_np = adversarial_image.permute(1, 2, 0).cpu().numpy()
    adversarial_image_np = (adversarial_image_np * 255).astype("uint8")
    adversarial_pil_image = Image.fromarray(adversarial_image_np)
    adversarial_pil_image.save(ouput_image_path)


def verify_similarity(image_path, query):
    clip_vision_model.eval()
    clip_text_model.eval()

    image = Image.open(image_path).convert("RGB")
    image_inputs = clip_vision_processor(images=image, return_tensors="pt").to("cuda")

    with torch.no_grad():
        image_outputs = clip_vision_model(**image_inputs)
    image_embeds = image_outputs.image_embeds 
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

    # Tokenize and obtain text embeddings
    text_inputs = clip_tokenizer([query], return_tensors="pt").to("cuda")
    with torch.no_grad():
        text_outputs = clip_text_model(**text_inputs)
    text_embeds = text_outputs.text_embeds  
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    # Compute cosine similarity
    similarity = F.cosine_similarity(image_embeds, text_embeds)
    similarity_score = similarity.item()
    return similarity_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="MMQA", help=["MMQA", "WebQA"])
    parser.add_argument("--metadata_path", type=str, help="path to lpa-bb poisoned metadata file")
    parser.add_argument("--save_data_dir", type=str, default='./results', help="save dir path for metadata")
    parser.add_argument("--save_img_dir", type=str, default='./results', help="save dir path for poisoned images")
    parser.add_argument("--num_steps", type=int, default=50, help="the number of advesarial optimization")
    parser.add_argument("--eps", type=float, default=0.05, help="attack strength")
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
    args = parser.parse_args()
    
    os.makedirs(args.save_data_dir, exist_ok=True)
    os.makedirs(args.save_img_dir, exist_ok=True)

    with open(args.metadata_path, 'r') as file:
        metadata = json.load(file)
        
    target_answers, poisoned_img_paths, poison_img_captions, qids = [], [], [], []
    gt_answers, questions = [], []
    for index, item in tqdm(enumerate(metadata)):
        image_path = item['poisoned_img_path']
        query = item['question']
        ouput_image_path = f"{args.save_img_dir}/{item['qid']}.png"
        poisoned_img_paths.append(ouput_image_path)
        target_answers.append(item["wrong_answer"])
        qids.append(item['qid'])
        gt_answers.append(item["gt_answer"])
        questions.append(query)
        poison_img_captions.append(item["poisoned_caption"])
        generate_adversarial_image(
            image_path,
            query,
            ouput_image_path,
            clip_vision_model,
            clip_text_model,
            clip_tokenizer,
            num_steps=args.num_steps,
            epsilon=args.eps,
            alpha=args.lr,
        )
        original_similarity = verify_similarity(image_path, query)
        adversarial_similarity = verify_similarity(ouput_image_path, query)
        print(f"Case: {index}, Original Similarity: {original_similarity}, Adversarial Similarity: {adversarial_similarity}")

    build_metadata(
        args.task, args.metadata_path, args.save_data_dir, poisoned_img_paths, 
        target_answers=target_answers, gt_answers=gt_answers, questions=questions, poison_img_captions=poison_img_captions, poison_type='lpa-rt'
    )
