from .logger import RougeEvalLogger

from tqdm.contrib import tzip
from typing import List

from vllm import LLM, SamplingParams

def get_prefix_before_words_occur(string: str, words: List[str]) -> str:
    for word in words: string = string.split(word)[0]
    return string


def eval(
    model, tokenizer,
    questions: List[str], choices: List[List[str]], answers: List[str],
    icl_prompt: str = "",
    max_new_tokens : int = 16
):
    if isinstance(model, LLM):
        return _eval_vllm(
            model, tokenizer,
            questions, choices, answers,
            icl_prompt,
            max_new_tokens=max_new_tokens
        )
    else:
        return _eval(
            model, tokenizer,
            questions, choices, answers,
            icl_prompt,
            max_new_tokens=max_new_tokens
        )


def _eval(
    model, tokenizer,
    questions: List[str], choices: List[List[str]], answers: List[str],
    icl_prompt: str = "",
    max_new_tokens : int = 32
):
    raise NotImplementedError


def _eval_vllm(
    model, tokenizer,
    questions: List[str], choices: List[List[str]], answers: List[str],
    icl_prompt: str = "",
    max_new_tokens : int = 32
):
    assert len(questions) == len(answers)
    answers = list(map(int, answers))
    #logger = RougeEvalLogger()

    prompts = []
    letters = ["A", "B", "C", "D"]
    for q, c, a in zip(questions, choices, answers):
        prompt = icl_prompt + "\n" + \
            f"Quesion: {q}\n\n" + \
            f"Choices:\n" + \
            "\n".join(
                [
                    f"{letter}. {answer}" 
                    for answer, letter in zip(c, letters)
                ]
            ) + \
            f"\n\nCorrect answer:"
        prompts.append(prompt)

    sp = SamplingParams(temperature=0, max_tokens=max_new_tokens)
    outputs = model.generate(prompts, sp)
    outputs = [output.outputs[0].text for output in outputs]

    def parse_output(output):
        output = get_prefix_before_words_occur(output, ["\n\n", "\nQuestion", "Question:"])
        return output.strip()

    outputs = list(map(parse_output, outputs))

    def extract_answer(output):
        try:
            index = letters.index(output[0])
        except:
            index = -1
        finally:
            return index

    pred = list(map(extract_answer, outputs))
    acc = sum([int(a == b) for a, b in zip(answers, pred)]) / len(answers)

    #for question, answer, output in zip(questions, answers, outputs):
    #    output = get_prefix_before_words_occur(output, ["\n\n", "\nQuestion", "Question:"])
    #    logger.log(prompt, letters[int(answer)], output, question=question)

    return acc
