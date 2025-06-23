from authentication.authentication import hf_login
from audio.conversion import change_audio_format
from audio.chunking import ChunkedAudio, ChunkedText
from audio.transcription import convert_to_text
from audio.transcription import transcribable_formats
from dotenv import load_dotenv
import yaml
import os
from pathlib import Path
from typing import Dict
import transformers

path_to_config = "processing/config/CONFIG.yaml"
path_to_inputs = "processing/inputs"
path_to_raw_outputs = "processing/outputs/raw"
path_to_refined_outputs = "processing/outputs/refined"


def transcribe_audio_file(audio_file: Path, pipe: transformers.pipeline, config: Dict) -> transformers.pipeline:
    # Changing the format to a transcribable one
    if audio_file.suffix not in transcribable_formats:
        audio_file = change_audio_format(audio_file, "mp3")

    preparation_time = 0
    transcription_time = 0

    input_file = Path(audio_file)
    base_path = audio_file.parent.parent
    output_name = input_file.stem + "_raw_transcript.txt"  # stem = file_name without extension
    output_file = base_path / "outputs" / output_name

    if "chunk_window" in config.keys():
        print(f"Converting: {audio_file} ... \nIt will be processed split in chunks")
        # Chunking the audio
        chunked_audio = ChunkedAudio(audio_file, config["chunk_window"])
        chunked_audio.chunk_resource()
        chunks = chunked_audio.chunks_paths

        chunked_text = ChunkedText(output_file)
        for chunk in chunks:
            print(f"Converting chunk: {chunk}")
            # Converting audio into text
            response = convert_to_text(audio_file, config, pipe)
            pipe = response["pipeline"]

            preparation_time += response["preparation_time"]
            transcription_time += response["transcription_time"]

            base_path = chunk.parent.parent
            output_chunk_name = chunk.stem + "_raw_transcript.txt"
            output_chunk = base_path / "outputs" / output_chunk_name

            output_chunk.write_text(response["transcript"]["text"].strip(), encoding="utf-8")
            print(f"Chunk {chunk} converted and stored in {output_chunk}")

            # Tracking the chunked text
            chunked_text.append_chunk(chunk)

        # Merging the chunks
        chunked_text.merge_chunks()
        # Deleting the chunks
        chunked_audio.delete_chunks()
        chunked_text.delete_chunks()
    else:
        print(f"Converting: {audio_file} ... \nIt will be processed as a whole")
        # Converting audio into text
        response = convert_to_text(audio_file, config, pipe)
        pipe = response["pipeline"]
        preparation_time = response["preparation_time"]
        transcription_time = response["transcription_time"]
        output_file.write_text(response["transcript"]["text"].strip(), encoding="utf-8")

    print("###########################################################################")
    print(f"{audio_file} converted and stored in {output_name}!")
    print(f"Preparation time: {preparation_time}")
    print(f"Transcription time: {transcription_time}")
    print("###########################################################################")

    return pipe


def refine_text_file(text_file: Path) -> transformers.pipeline:
    pass


if __name__ == "__main__":
    # Loading the configuration file
    with open(path_to_config, "r") as f:
        config = yaml.safe_load(f)

    # Automatically loading from .env if present in root
    load_dotenv()
    # Logging in HuggingFace
    hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        raise ValueError("HF_TOKEN not found in environment variables.")
    hf_login(hf_token)

    # Iterating over the contained in the inputs folder, getting the transcript of each audio file contained in the
    # inputs directory
    transcription_pipeline = None
    inputs_folder = Path(path_to_inputs)
    for file in inputs_folder.iterdir():
        if file.is_file():
            transcription_pipeline = transcribe_audio_file(file, transcription_pipeline, config)
    print("Transcription routine ended")

    # Iterating over the contained in the raw outputs folder, getting the refined text of each text file contained in
    # the raw outputs directory
    refinement_pipeline = None
    raw_inputs_folder = Path(path_to_raw_outputs)
    if "refiner" in config.keys() and config["refine"]:
        for file in raw_inputs_folder.iterdir():
            if file.is_file():
                refinement_pipeline = refine_text_file(file)
        print("Refinement routine ended")
