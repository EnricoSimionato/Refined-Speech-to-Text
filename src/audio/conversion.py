from moviepy import AudioFileClip, VideoFileClip
from pathlib import Path


def change_audio_format(file_path: str, output_extension: str = "mp3") -> str:
    """
    Changes into a chose audio format the input file.
    If the input is a video, it extracts only the audio.

    Parameters:
         file_path (str): Path to the audio file.
         output_extension (str, optional): Extension of the output file.

    Returns:
        str: Path to the output file.
    """
    print(f"Converting {file_path} to a {output_extension} file ...")

    video_formats = [
        ".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv", ".mpeg", ".mpg", ".ogv", ".3gp", ".ts", ".m4v"
    ]
    audio_formats = [
        ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus", ".wma", ".alac", ".aiff"
    ]
    if output_extension not in audio_formats and "." + output_extension not in audio_formats:
        raise TypeError('The output extension must be either .mp3 or .wav')

    input_path = Path(file_path)
    ext = input_path.suffix.lower() # Extension (.extension)

    # If the extension is already the desired one, the function returns
    if output_extension == ext and "." + output_extension == ext:
        print(f"The output file {input_path} is already a {output_extension} file.")
        return

    # Defining the output path
    new_file_name = input_path.stem + f".{output_extension}"
    output_path = input_path.parent / new_file_name

    if output_path == input_path:
        print("There is already a file with the same name and extension.")
        return

    # Loading appropriate clip type
    if ext in video_formats:
        clip = VideoFileClip(input_path)
        audio = clip.audio
    elif ext in audio_formats:
        audio = AudioFileClip(input_path)
    else:
        raise ValueError(f"Unsupported input format: {ext}")

    # Writing the output audio file
    audio.write_audiofile(output_path)

    return output_path
