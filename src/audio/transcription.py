#from exporch import get_available_device
import time
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


convertible_formats = [".mp3"]


def convert_to_text(audio_name: str, config: dict, pipe = None) -> dict:
    """
    Converts the audio file to its textual transcription.

    Parameters:
        audio_name (str): The path of the audio file.
        config (dict): The configuration of the application containing additional parameters.
        pipe (transformers.pipeline): The pipeline to use, if already instantiated.

    Returns:
        dict: The transcription of the audio file.
    """

    preparation_time = 0
    if not pipe:
        torch_dtype = torch.float32
        batch_size = config["batch_size"] if "batch_size" in config.keys() else None
        chunk_length_s = config["chunk_length_s"] if "chunk_length_s" in config.keys() else None
        #device = get_available_device(config["device"] if "device" in config.keys() else "cuda")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = config["model_id"] if "model_id" in config.keys() else "openai/whisper-large-v3"

        init_loading = time.time()
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )

        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        # If chunk_length_s and batch_size are present, batch mode is used (faster inference - lower accuracy)
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=chunk_length_s,
            batch_size=batch_size,
            torch_dtype=torch_dtype,
            device=device,
        )

        end_loading = time.time()
        preparation_time = end_loading - init_loading

        print("Pipeline ready")

    init_transcription = time.time()
    transcript = pipe(str(audio_name), return_timestamps=True)
    end_transcription = time.time()
    transcription_time = end_transcription - init_transcription

    return {
        "pipeline": pipe,
        "transcript": transcript,
        "preparation_time": preparation_time,
        "transcription_time": transcription_time
    }
