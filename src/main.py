from authentication.authentication import hf_login
from audio.conversion import change_audio_format
from audio.chunking import ChunkedAudio, ChunkedText
from audio.transcription import convert_to_text
from audio.transcription import convertible_formats
from dotenv import load_dotenv
import yaml
import os
from pathlib import Path


path_to_config = "processing/config/CONFIG.yaml"
path_to_inputs = "processing/inputs"
path_to_outputs = "processing/outputs"


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

    pipe = None

    # Iterating over the contained in the input folder, obtaining the transcript of each audio file contained in the
    # inputs directory
    folder = Path(path_to_inputs)
    for file in folder.iterdir():
        if file.is_file():
            if file.suffix not in convertible_formats:
                file = change_audio_format(file, "mp3")
            preparation_time = 0
            transcription_time = 0

            input_file = Path(file)
            base_path = file.parent.parent
            output_name = input_file.stem + "_raw_transcript.txt"  # stem = file_name without extension
            output_file = base_path / "outputs" / output_name

            if "chunk_window" in config.keys():
                print(f"{file} will be processed split in chunks")
                # Chunking the audio
                chunked_audio = ChunkedAudio(file, config["chunk_window"])
                chunked_audio.chunk_resource()
                chunks = chunked_audio.chunks_paths

                chunked_text = ChunkedText(output_file)
                for chunk in chunks:
                    print(f"Converting chunk: {chunk}")
                    response = convert_to_text(file, config, pipe)
                    pipe = response["pipeline"]
                    print(f"Chunk {chunk} converted")
                    preparation_time += response["preparation_time"]
                    transcription_time += response["transcription_time"]

                    base_path = chunk.parent.parent
                    output_chunk_name = chunk.stem + "_raw_transcript.txt"
                    output_chunk = base_path / "outputs" / output_chunk_name

                    output_chunk.write_text(response["transcript"]["text"].strip(), encoding="utf-8")

                    # Tracking the chunked text
                    chunked_text.append_chunk(chunk)

                # Merging the chunks
                chunked_text.merge_chunks()
                # Deleting the chunks
                chunked_audio.delete_chunks()
                chunked_text.delete_chunks()
            else:
                print(f"{file} will be processed as a whole")
                print(f"Converting: {file}")
                response = convert_to_text(file, config, pipe)
                pipe = response["pipeline"]
                preparation_time = response["preparation_time"]
                transcription_time = response["transcription_time"]
                print(f"{file} converted")
                output_file.write_text(response["transcript"]["text"].strip(), encoding="utf-8")

            print(f"Preparation time: {preparation_time}")
            print(f"Transcription time: {transcription_time}")
