from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from peft import AutoPeftModelForCausalLM
import torch


def qwen_eval_relevance(image_path, question, model, tokenizer):

    query_list = [{"image": image_path}]

    query_list.append({"text": question})

    query = tokenizer.from_list_format(query_list)
    outputs = model.chat(
        tokenizer,
        query=query,
        history=None,
        return_dict_in_generate=True,
        output_scores=True,
        do_sample=False,
    )
    logits = outputs.scores[0][0]

    probs = (
        torch.nn.functional.softmax(
            torch.FloatTensor(
                [
                    logits[tokenizer("Yes").input_ids[0]],
                    logits[tokenizer("No").input_ids[0]],
                ]
            ),
            dim=0,
        )
        .detach()
        .cpu()
        .numpy()
    )

    return probs[0]


def qwen_chat(image_path, question, model, tokenizer):

    query_list = []
    if image_path:
        # for img in image_path.split(","):
        for img in image_path:
            query_list.append({"image": img})

    query_list.append({"text": question})

    query = tokenizer.from_list_format(query_list)
    response, _ = model.chat(tokenizer, query=query, history=None, do_sample=False)

    return response

