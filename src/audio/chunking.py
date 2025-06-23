from pathlib import Path
from typing import List, override, Union
import ffmpeg

class ChunkedResource:
    """
    Chunked resource.
    """
    def __init__(self, path: str | Path, chunks_paths: List[str | Path] = None):
        self.path = path
        self.chunks_paths = chunks_paths if chunks_paths else []

    def chunk_resource(self, *args, **kwargs) -> None:
        """
        Chunks the resource.
        """

        pass

    def merge_chunks(self, *args, **kwargs) -> None:
        """
        Merges the chunks.
        """

        pass

    def append_chunk(self, chunk_path) -> None:
        """
        Appends the chunk to the list of chunks.

        Args:
            chunk_path: Path to the chunk file.
        """

        self.chunks_paths.append(chunk_path)

    def delete_chunks(self) -> None:
        """
        Deletes all the chunks.
        """

        for chunk_path in self.chunks_paths:
            try:
                chunk_path.unlink()
                print(f"Deleted chunk: {chunk_path}")
            except FileNotFoundError:
                print(f"Chunk already deleted: {chunk_path}")
            except Exception as e:
                print(f"Failed to delete {chunk_path}: {e}")

        # Clearing the list of chunks so it's no longer pointing to deleted files
        self.chunks_paths = []

class ChunkedAudio(ChunkedResource):
    """
    Chunked audio resource using ffmpeg (compatible with Python 3.13).
    """

    def __init__(
        self,
        path: Union[str, Path],
        chunks_paths: List[Union[str, Path]] = None,
        chunk_length_s: int = 120,
        output_format: str = "mp3"
    ):
        if chunk_length_s <= 0:
            raise ValueError("Chunk length must be greater than 0.")
        super().__init__(path, chunks_paths)
        self.chunk_length_s = chunk_length_s
        self.output_format = output_format

    @override
    def chunk_resource(self):
        """
        Splits an audio file into chunks of specified length using ffmpeg.
        """
        input_path = Path(self.path)
        output_dir = input_path.parent

        # Get the duration of the file using ffmpeg.probe
        probe = ffmpeg.probe(str(input_path))
        duration = float(probe["format"]["duration"])
        total_chunks = int(duration // self.chunk_length_s) + int(duration % self.chunk_length_s != 0)
        num_digits = len(str(total_chunks - 1))

        chunk_paths = []

        for i in range(total_chunks):
            start = i * self.chunk_length_s
            out_file = output_dir / f"{input_path.stem}_chunk_{i:0{num_digits}d}.{self.output_format}"

            (
                ffmpeg
                .input(str(input_path), ss=start, t=self.chunk_length_s)
                .output(str(out_file), format=self.output_format)
                .overwrite_output()
                .run(quiet=True)
            )

            chunk_paths.append(out_file)

        self.chunks_paths = chunk_paths

class ChunkedText(ChunkedResource):
    def __init__(
            self,
            path: str | Path,
            chunks_paths: List[str | Path] = None,
            chunk_length_char: int = 100
    ):
        super().__init__(path, chunks_paths)
        self.chunk_length_char = chunk_length_char

    def merge_chunks(self, *args, **kwargs) -> None:
        """
        Merges all text chunk files into one single text file and saves it at `self.path`.
        """

        merged_text = ""
        for chunk_path in sorted(self.chunks_paths):
            if chunk_path.exists() and chunk_path.suffix == ".txt":
                text = chunk_path.read_text(encoding="utf-8")
                merged_text += text.strip() + "\n\n"
            else:
                print(f"Skipping invalid or missing chunk: {chunk_path}")

        # Saving the merged content
        self.path.write_text(merged_text.strip(), encoding="utf-8")
        print(f"Merged text saved to: {self.path}")

