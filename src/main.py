from authentication.authentication import hf_login
from audio.conversion import change_audio_format
from audio.chunking import ChunkedAudio, ChunkedText
from audio.transcription import convert_to_text
from audio.transcription import transcribable_formats
from dotenv import load_dotenv
import yaml
import os
from pathlib import Path
from language.refinement import refine_text
from typing import Dict, List
import transformers

path_to_config = Path("processing/config/CONFIG.yaml")
path_to_inputs = Path("processing/inputs")
path_to_raw_outputs = Path("processing/outputs/raw")
path_to_refined_outputs = Path("processing/outputs/refined")


def transcribe_audio_file(audio_file: Path, config: Dict, pipe: transformers.pipeline) -> transformers.pipeline:
    # Changing the format to a transcribable one
    if audio_file.suffix not in transcribable_formats:
        audio_file = change_audio_format(audio_file, "mp3")

    preparation_time = 0
    transcription_time = 0

    output_name = audio_file.stem + "_raw_transcript.txt"  # stem = file_name without extension
    output_file = path_to_raw_outputs / output_name

    # TODO Chunking needs to be tested
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

            output_chunk_name = chunk.stem + "_raw_transcript.txt"
            output_chunk = path_to_raw_outputs / output_chunk_name

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
    print(f"{audio_file} converted and stored in {output_file}!")
    print(f"Preparation time: {preparation_time}")
    print(f"Transcription time: {transcription_time}")
    print("###########################################################################")

    return pipe


def refine_text_file(text_file: Path, config: Dict, pipelines = None) -> transformers.pipeline:
    output_name = Path(str(text_file.stem).replace("_raw_transcript", "refined_transcript") + ".txt")
    output_file = path_to_refined_outputs / output_name

    print(f"Refining: {text_file} ...")
    # Converting audio into text
    response = refine_text(text_file, config, *pipelines)

    language_detection_pipeline = response["language_detection_pipeline"]
    translator_pipeline = response["translator_pipeline"]
    language_translator = response["language_translator"]
    refining_pipeline = response["refined_pipeline"]

    preparation_time = response["preparation_time"]
    transcription_time = response["transcription_time"]
    output_file.write_text(response["transcript"]["text"].strip(), encoding="utf-8")

    print("###########################################################################")
    print(f"{text_file} refined and stored in {output_file}!")
    print(f"Preparation time: {preparation_time}")
    print(f"Transcription time: {transcription_time}")
    print("###########################################################################")

    return language_detection_pipeline, translator_pipeline, language_translator, refining_pipeline


if __name__ == "__main__":
    # Loading the configuration file
    with open(path_to_config, "r") as f:
        configuration = yaml.safe_load(f)

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
    for file in path_to_inputs.iterdir():
        if file.is_file():
            transcription_pipeline = transcribe_audio_file(file, configuration, transcription_pipeline)
    print("Transcription routine ended")

    # Iterating over the contained in the raw outputs folder, getting the refined text of each text file contained in
    # the raw outputs directory
    pipelines = None
    if "refine" in configuration.keys() and configuration["refine"]:
        for file in path_to_raw_outputs.iterdir():
            if file.is_file():
                pipelines = refine_text(file, configuration, pipelines)
        print("Refinement routine ended")
