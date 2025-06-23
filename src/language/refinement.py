from pathlib import Path
import time
from typing import Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available



def refine_text(
        text_file: Path,
        config: dict,
        model: AutoModelForCausalLM = None,
        tokenizer: AutoTokenizer = None
) -> dict:
    """
    Refines the raw text file obtained from the audio transcription into a cleaner text file.

    Parameters:
        text_file (Path): Path to the raw audio transcription file.
        config (dict): The configuration of the application containing additional parameters.
        model (AutoModelForCausalLM, optional): The model to use.
        tokenizer (AutoTokenizer, optional): The tokenizer to use.

    Returns:
        dict: The refined text fine,
            the pipeline refining it,
            the preparation time of the pipeline,
            the refinement time of the raw text.

        >>  {
        >>      "pipeline": pipe,
        >>      "transcript": transcript,
        >>      "preparation_time": preparation_time,
        >>      "transcription_time": transcription_time
        >>  }
    }
    """

    preparation_time = 0
    if not model:
        # TODO Loading can use exporch but it does not work with python 3.11 on colab
        # Loading the model and the tokenizer
        init_loading = time.time()
        model, tokenizer = load_model(config)
        end_loading = time.time()
        preparation_time = end_loading - init_loading
        print("Pipeline ready")

    # Loading the raw transcript
    raw_transcript = Path(text_file).read_text()

    # Defining the prompt
    prompt = (
        "### Instruction:\n"
        "Improve the following transcription by removing filler words, errors in the transcription or resulting from "
        "background music or noise, fixing grammar, and making it fluent. Keep unaltered the concepts in the text and "
        "use the same language used in the text. TEXT TO REFINE:\n\n"
        f"{raw_transcript}\n\n### Response:"
    )

    # Tokenizing the input
    init_refinement = time.time()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Computing the refined transcript
    refined_transcript = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    end_refinement = time.time()
    refinement_time = end_refinement - init_refinement

    return {
        "model": model,
        "tokenizer": tokenizer,
        "refined_transcript": tokenizer.decode(refined_transcript[0], skip_special_tokens=True),
        "preparation_time": preparation_time,
        "transcription_time": refinement_time
    }

def load_model(
        config: dict
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Loads a model with the given quantization and device settings.

    Parameters:
        config (dict): The configuration of the application containing additional parameters.

    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: The model and the tokenizer
    """

    if "model_id_refiner" not in config.keys():
        raise KeyError("Model ID refiner not found in configuration")

    model_id = config["model_id_refiner"]
    quantization = config["quantization_refiner"] if "quantization_refiner" in config.keys() else None
    float16 = config["float16_refiner"] if "float16_refiner" in config.keys() else None
    device = config["device"] if "device" in config.keys() else "auto"
    # TODO fix the problem with exporch and python 3.11
    # device = get_available_device(config["device"] if "device" in config.keys() else "cuda")
    use_flash_attention = config["use_flash_attention"] if "use_flash_attention" in config.keys() else False

    # Loading tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Defining the quantization configuration
    quant_config = None
    if quantization in {"4bit", "8bit"}:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=(quantization == "4bit"),
            load_in_8bit=(quantization == "8bit"),
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

    # Loading the model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device,
        quantization_config=quant_config,
        torch_dtype=torch.float16 if float16 and quant_config is None else "auto",
        trust_remote_code=True,
    )

    # Using flash attention if available and requested
    if use_flash_attention and is_flash_attn_2_available():
        try:
            model = model.to(memory_format=torch.channels_last)
        except Exception as e:
            print("Flash attention setup failed:", e)

    return tokenizer, model

if __name__ == "__main__":
    config = {
        "device": "cuda",
        "quantization_refiner": "8bits",
        "model_id_refiner": "meta-llama/Llama-3.1-8B-Instruct",
        "model_id_transcriber": "openai/whisper-large-v3",
        "refine": True,
        #"use_flash_attention": true
    }
    resp = refine_text(Path("text.txt"), config, None, None)
    print(resp[["refined_transcript"]])
