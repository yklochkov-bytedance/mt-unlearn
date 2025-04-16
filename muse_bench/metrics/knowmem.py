from .logger import RougeEvalLogger

from tqdm.contrib import tzip
from typing import List

from vllm import LLM, SamplingParams

def get_prefix_before_words_occur(string: str, words: List[str]) -> str:
    for word in words: string = string.split(word)[0]
    return string


def eval(
    model, tokenizer,
    questions: List[str], answers: List[str],
    icl_qs: List[str] = [], icl_as: List[str] = [],
    max_new_tokens : int = 32
):
    if isinstance(model, LLM):
        return _eval_vllm(
            model, tokenizer,
            questions, answers,
            icl_qs, icl_as,
            max_new_tokens=max_new_tokens
        )
    else:
        return _eval(
            model, tokenizer,
            questions, answers,
            icl_qs, icl_as,
            max_new_tokens=max_new_tokens
        )


def _eval(
    model, tokenizer,
    questions: List[str], answers: List[str],
    icl_qs: List[str] = [], icl_as: List[str] = [],
    max_new_tokens : int = 32
):
    assert len(questions) == len(answers)
    assert len(icl_qs) == len(icl_as)

    logger = RougeEvalLogger()
    general_prompt: str = ""

    # Few-shot prompting
    for question, answer in zip(icl_qs, icl_as):
        general_prompt += f"Question: {question}\nAnswer: {answer}\n\n"

    for question, answer in tzip(questions, answers):
        prompt = general_prompt + f"Question: {question}\nAnswer: "

        # Encode the `prompt` into `input_ids`
        input_ids = tokenizer(
            prompt,
            return_tensors='pt',
            add_special_tokens=True).input_ids

        # Use the `model` to generate the continuation of the `input_ids`.
        output_ids = model.generate(
            input_ids.to(model.device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id)
        output_ids = output_ids[:, len(input_ids[0]):]

        output = tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True)[0]

        output = get_prefix_before_words_occur(output, ["\n\n", "\nQuestion", "Question:"])
        logger.log(prompt, answer, output, question=question)

    return logger.report()



def _eval_vllm(
    model, tokenizer,
    questions: List[str], answers: List[str],
    icl_qs: List[str] = [], icl_as: List[str] = [],
    max_new_tokens : int = 32
):
    assert len(questions) == len(answers)
    assert len(icl_qs) == len(icl_as)

    logger = RougeEvalLogger()
    general_prompt: str = ""

    # Few-shot prompting
    for question, answer in zip(icl_qs, icl_as):
        general_prompt += f"Question: {question}\nAnswer: {answer}\n\n"

    prompts = []
    for question, answer in zip(questions, answers):
        prompt = general_prompt + f"Question: {question}\nAnswer: "
        prompts.append(prompt)

    sp = SamplingParams(temperature=0, max_tokens=max_new_tokens)
    outputs = model.generate(prompts, sp)
    outputs = [output.outputs[0].text for output in outputs]

    for prompt, question, answer, output in zip(prompts, questions, answers, outputs):
        output = get_prefix_before_words_occur(output, ["\n\n", "\nQuestion", "Question:"])
        logger.log(prompt, answer, output, question=question)

    return logger.report()
