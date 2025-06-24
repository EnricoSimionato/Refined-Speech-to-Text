from exporch import get_available_device
from pathlib import Path
import time
from typing import Tuple, Dict
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Gemma3ForCausalLM, pipeline
from transformers.utils import is_flash_attn_2_available


def refine_text(
        text_file: Path,
        config: dict,
        refinement_pipeline: transformers.pipeline = None,
        language_detection_pipeline: transformers.pipeline = None
) -> Dict:
    preparation_time = 0
    if not refinement_pipeline:
        model_id = config["model_id_refiner"] if "model_id_refiner" in config.keys() else "google/gemma-3-1b-it"
        device = get_available_device(config["device"] if "device" in config.keys() else "cuda")

        # Loading the refinement pipeline
        init_loading = time.time()
        refinement_pipeline = pipeline("text-generation", model=model_id, device=device, torch_dtype=torch.bfloat16)
        end_loading = time.time()
        preparation_time += end_loading - init_loading
        print("Refinement pipeline ready")

    if language_detection_pipeline is None:
        # Loading the language detection pipeline
        init_loading = time.time()
        model_ckpt = "papluca/xlm-roberta-base-language-detection"
        language_detection_pipeline = pipeline("text-classification", model=model_ckpt)
        end_loading = time.time()
        preparation_time += end_loading - init_loading
        print("Language detection pipeline ready")

    # Loading the raw transcript
    raw_transcript = Path(text_file).read_text()

    # Detecting the language
    init_refinement = time.time()
    lang = language_detection_pipeline(raw_transcript, top_k=1, truncation=True)[0]["label"]
    translator_pipeline = pipeline(f"translation_en_to_{lang}", model=f"Helsinki-NLP/opus-mt-en-{lang}")

    model = refinement_pipeline.model
    max_context_length = max(
        getattr(model.config, "max_position_embeddings", 0),
        refinement_pipeline.tokenizer.model_max_length
    )

    # Defining the prompt to input to the model
    refinement_marker = "<RT>"
    prompt = (
        "Improve the following transcription by removing filler words, errors in the transcription or resulting from "
        "background music or noise, correcting grammar, and making it fluent. Keep unaltered the concepts in the text."
        f"Return just the refined transcription writing it after the tag {refinement_marker}."
    )

    # Translating the prompt into the language of the text
    translated_prompt = translator_pipeline(prompt)[0]["translation_text"]
    translated_system_prompt = translator_pipeline("You are an assistant that helps refining texts.")[0]["translation_text"]
    messages = [
        [
            {"role": "system", "content": [{"type": "text", "text": translated_system_prompt},]},
            {"role": "user", "content": [{"type": "text", "text": translated_prompt+"\n\n"+f"{raw_transcript}"},]},
        ],
    ]

    # Refining the text
    refined_transcript = refinement_pipeline(messages, max_new_tokens=10000)[0][0]["generated_text"][-1]["content"]
    parts = refined_transcript.split("\n\n", 1)
    refined_transcript = parts[1].strip() if len(parts) > 1 else refined_transcript
    end_refinement = time.time()
    refinement_time = end_refinement - init_refinement

    return {
        "language_detection_pipeline": language_detection_pipeline,
        "refined_pipeline": refinement_pipeline,
        "refined_transcript": refined_transcript,
        "preparation_time": preparation_time,
        "transcription_time": refinement_time
    }



def refine_text_using_model(
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
    if not model or not tokenizer:
        # TODO Loading can use exporch but it does not work with python 3.11 on colab
        # Loading the model and the tokenizer
        init_loading = time.time()
        #model, tokenizer = load_model(config)
        model_id = "google/gemma-3-1b-it"

        model = Gemma3ForCausalLM.from_pretrained(
            model_id
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        end_loading = time.time()
        preparation_time = end_loading - init_loading
        print("Pipeline ready")

    # Loading the raw transcript
    raw_transcript = Path(text_file).read_text()

    # Defining the prompt to input to the model
    prompt = ("Improve the following transcription by removing filler words, errors in the transcription or resulting "
              "from background music or noise, fixing grammar, and making it fluent. Keep unaltered the concepts in the "
              f"text and use the same language used in the text. TEXT TO REFINE:\n\n"f"{raw_transcript}\n\n")
    messages = [
        [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}, ]},
            {"role": "user", "content": [{"type": "text", "text": prompt}, ]},
        ],
    ]
    init_refinement = time.time()
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device).to(torch.bfloat16)

    # Generating the answer
    with torch.inference_mode():
        print(len(inputs["input_ids"]))
        print(len(inputs["input_ids"][0]))
        print(len(inputs["input_ids"][1]))
        outputs = model.generate(**inputs, max_new_tokens=len(inputs["input_ids"]))

    # Decoding the answer
    refined_transcript = tokenizer.batch_decode(outputs)
    end_refinement = time.time()
    refinement_time = end_refinement - init_refinement

    """
    # Tokenizing the input
    init_refinement = time.time()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Computing the refined transcript
    refined_transcript = model.generate(
        **inputs,
        max_new_tokens=len(raw_transcript),
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    end_refinement = time.time()
    refinement_time = end_refinement - init_refinement
    """

    return {
        "model": model,
        "tokenizer": tokenizer,
        "refined_transcript": refined_transcript,#tokenizer.decode(refined_transcript[0], skip_special_tokens=True)[len(prompt):],
        "preparation_time": preparation_time,
        "transcription_time": refinement_time
    }

#
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
        low_cpu_mem_usage=True
    )

    # Using flash attention if available and requested
    if use_flash_attention and is_flash_attn_2_available():
        try:
            model = model.to(memory_format=torch.channels_last)
        except Exception as e:
            print("Flash attention setup failed:", e)

    return model, tokenizer

if __name__ == "__main__":
    config = {
        "device": "mps",
        #"quantization_refiner": "8bit",
        "float16_refiner": True,
        "model_id_refiner": "google/gemma-3-4b-it",
        "model_id_transcriber": "openai/whisper-large-v3",
        "refine": True,
        #"use_flash_attention": true
    }
    test_file = Path("test.txt")
    resp = refine_text(test_file, config, None)

    print(resp["refined_transcript"])