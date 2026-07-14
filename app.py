import atexit
import base64
import copy
import os
import shutil
import subprocess
import tempfile
import time
import traceback
import uuid
from io import BytesIO
from pathlib import Path

import captacity
import gradio as gr
import numpy
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from moviepy.editor import ImageClip
from moviepy.editor import CompositeVideoClip, VideoFileClip


APP_TEMP_ROOT = Path(tempfile.mkdtemp(prefix="captacity_gradio_"))
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv"}
DEFAULT_OUTPUT_NAME = "captioned_output.mp4"
RENDER_OUTPUT_NAME = "captioned_render.mp4"
TEXT_RENDER_PADDING_RATIO = 0.35
FONT_CHOICES = {
    "Bangers": "Bangers-Regular.ttf",
    "Knewave": "Knewave-Regular.ttf",
    "Poetsen One": "PoetsenOne-Regular.ttf",
    "Urbanist": "assets/fonts/Urbanist-Bold.ttf",
    "Coplette": "assets/fonts/Coplette.otf",
}
CAPTION_TYPE_COLOR = "Color Highlight"
CAPTION_TYPE_BOX = "Box Highlight"
CAPTION_TYPES = [CAPTION_TYPE_COLOR, CAPTION_TYPE_BOX]


def cleanup_temp_root() -> None:
    shutil.rmtree(APP_TEMP_ROOT, ignore_errors=True)


atexit.register(cleanup_temp_root)


def check_ffmpeg() -> str:
    if shutil.which("ffmpeg"):
        return ""

    return (
        "ffmpeg was not found on PATH. Install it before rendering captions:\n\n"
        "- macOS: `brew install ffmpeg`\n"
        "- Ubuntu/Debian: `sudo apt install ffmpeg`\n"
        "- Windows: `choco install ffmpeg` or download it from https://ffmpeg.org"
    )


def resolve_font(font_choice: str) -> str:
    font = FONT_CHOICES.get(font_choice, "Bangers-Regular.ttf")
    font_path = Path(font)
    if font_path.is_absolute() and font_path.exists():
        return str(font_path)

    bundled_path = Path(__file__).parent / font_path
    if bundled_path.exists():
        return str(bundled_path)

    return captacity.get_font_path(font)


def caption_type_preview(
    caption_type: str,
    font_type: str,
    font_size: int | float,
    word_highlight_color: str,
) -> str:
    try:
        preview_size = max(22, min(96, int(font_size) // 2))
    except (TypeError, ValueError):
        preview_size = 64

    image = render_line_image(
        words=["SHE", "KEPT", "ME", "UP"],
        current_index=1,
        first_word_index=0,
        caption_type=caption_type,
        font_size=preview_size,
        font_color="white",
        word_highlight_color=word_highlight_color,
        font=resolve_font(font_type),
        stroke_color="black",
        stroke_width=max(1, preview_size // 24),
        text_padding=max(12, preview_size // 3),
    )

    background_padding = 10
    background = Image.new(
        "RGBA",
        (
            image.width + background_padding * 2,
            image.height + background_padding * 2,
        ),
        (27, 27, 31, 255),
    )
    background.paste(image, (background_padding, background_padding), image)

    buffer = BytesIO()
    background.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return (
        '<div style="display:flex;align-items:center;padding:8px 0;">'
        f'<img alt="Caption preview" src="data:image/png;base64,{encoded}" '
        'style="max-width:100%;height:auto;border-radius:8px;" />'
        "</div>"
    )


def validate_video(video_path: str | None) -> Path:
    if not video_path:
        raise gr.Error("Upload a video before generating captions.")

    path = Path(video_path)
    suffix = path.suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise gr.Error(f"Unsupported video type '{suffix or 'unknown'}'. Use {allowed}.")

    if not path.exists():
        raise gr.Error("The uploaded video could not be found. Please upload it again.")

    return path


def copy_upload(video_path: Path, work_dir: Path) -> Path:
    input_path = work_dir / f"input{video_path.suffix.lower()}"
    shutil.copy2(video_path, input_path)
    return input_path


def extract_audio(video_file: str) -> str:
    temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    captacity.ffmpeg(["ffmpeg", "-y", "-i", video_file, temp_audio_file])
    return temp_audio_file


def transcribe_video(video_file: str) -> list[dict]:
    print("Extracting audio...")
    audio_file = extract_audio(video_file)
    print("Transcribing audio...")
    return captacity.transcriber.transcribe_locally(audio_file, None)


def segments_to_text(segments: list[dict]) -> str:
    words = []
    for segment in segments:
        for word in segment.get("words", []):
            words.append(word.get("word", "").strip())
    return " ".join(words).strip()


def edited_text_to_segments(edited_text: str, original_segments: list[dict]) -> list[dict]:
    edited_words = edited_text.split()
    if not edited_words:
        raise gr.Error("Transcript text is empty. Add text or clear it to transcribe again.")

    original_words = [
        word
        for segment in original_segments
        for word in segment.get("words", [])
    ]
    if not original_words:
        raise gr.Error("No word timestamps were found. Transcribe the video again.")

    segments = copy.deepcopy(original_segments)
    if len(edited_words) == len(original_words):
        word_index = 0
        for segment in segments:
            for word in segment.get("words", []):
                word["word"] = " " + edited_words[word_index]
                word_index += 1
        return segments

    start = float(original_words[0]["start"])
    end = float(original_words[-1]["end"])
    duration = max(0.01, end - start)
    step = duration / len(edited_words)
    redistributed_words = []
    for index, word_text in enumerate(edited_words):
        word_start = start + step * index
        word_end = start + step * (index + 1)
        redistributed_words.append(
            {
                "word": " " + word_text,
                "start": word_start,
                "end": word_end,
            }
        )

    return [
        {
            "start": redistributed_words[0]["start"],
            "end": redistributed_words[-1]["end"],
            "words": redistributed_words,
        }
    ]


def progress_status(percent: int, message: str) -> str:
    remaining = max(0, 100 - percent)
    return f"{percent}% complete, {remaining}% remaining - {message}"


def restore_original_audio(rendered_video: Path, input_video: Path, output_video: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(rendered_video),
        "-i",
        str(input_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(output_video),
    ]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffmpeg could not restore audio.")


def get_caption_y(
    video_height: int,
    text_height: int,
    top_padding: int,
    bottom_padding: int,
) -> int:
    centered_y = video_height // 2 - text_height // 2
    min_y = 0
    max_y = max(0, video_height - text_height)
    requested_y = centered_y + top_padding - bottom_padding
    return min(max(requested_y, min_y), max_y)


def render_line_image(
    words: list[str],
    current_index: int | None,
    first_word_index: int,
    caption_type: str,
    font_size: int,
    font_color: str,
    word_highlight_color: str,
    font: str,
    stroke_color: str,
    stroke_width: int,
    text_padding: int,
    fill_override: str | None = None,
) -> Image.Image:
    font_obj = ImageFont.truetype(font, font_size)
    full_text = " ".join(words)
    if caption_type == CAPTION_TYPE_BOX:
        full_text = full_text.upper()
    measure_image = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    draw = ImageDraw.Draw(measure_image)
    bbox = draw.textbbox((0, 0), full_text, font=font_obj, stroke_width=stroke_width)
    text_width = int(draw.textlength(full_text, font=font_obj))
    width = max(1, text_width + text_padding * 2 + stroke_width * 4)
    height = max(1, bbox[3] - bbox[1] + text_padding * 2)

    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    x = text_padding - min(0, bbox[0])
    y = text_padding - bbox[1]

    for local_index, word in enumerate(words):
        global_index = first_word_index + local_index
        highlighted = current_index is not None and global_index == current_index
        display_word = word.upper() if caption_type == CAPTION_TYPE_BOX else word
        color = fill_override or font_color
        if highlighted and caption_type == CAPTION_TYPE_COLOR:
            color = word_highlight_color

        if highlighted and caption_type == CAPTION_TYPE_BOX and fill_override is None:
            word_width = draw.textlength(display_word, font=font_obj)
            box_pad_x = max(6, int(font_size * 0.12))
            box_pad_y = max(3, int(font_size * 0.04))
            box_radius = max(4, int(font_size * 0.06))
            draw.rounded_rectangle(
                (
                    x - box_pad_x,
                    y + bbox[1] - box_pad_y,
                    x + word_width + box_pad_x,
                    y + bbox[3] + box_pad_y,
                ),
                radius=box_radius,
                fill=word_highlight_color,
            )

        draw.text(
            (x, y),
            display_word,
            font=font_obj,
            fill=color,
            stroke_width=stroke_width,
            stroke_fill=stroke_color,
        )
        x += draw.textlength(display_word + " ", font=font_obj)

    return image


def create_line_clip(
    words: list[str],
    current_index: int | None,
    first_word_index: int,
    caption_type: str,
    font_size: int,
    font_color: str,
    word_highlight_color: str,
    font: str,
    stroke_color: str,
    stroke_width: int,
    text_padding: int,
):
    image = render_line_image(
        words=words,
        current_index=current_index,
        first_word_index=first_word_index,
        caption_type=caption_type,
        font_size=font_size,
        font_color=font_color,
        word_highlight_color=word_highlight_color,
        font=font,
        stroke_color=stroke_color,
        stroke_width=stroke_width,
        text_padding=text_padding,
    )
    return ImageClip(numpy.array(image))


def create_line_shadow_clip(
    words: list[str],
    caption_type: str,
    font_size: int,
    font: str,
    stroke_width: int,
    text_padding: int,
    shadow_blur: float,
    opacity: float,
):
    image = render_line_image(
        words=words,
        current_index=None,
        first_word_index=0,
        caption_type=caption_type,
        font_size=font_size,
        font_color="black",
        word_highlight_color="black",
        font=font,
        stroke_color="black",
        stroke_width=stroke_width,
        text_padding=text_padding,
        fill_override="black",
    )
    image = image.filter(ImageFilter.GaussianBlur(radius=int(font_size * shadow_blur)))
    if opacity < 1:
        alpha = image.getchannel("A").point(lambda value: int(value * opacity))
        image.putalpha(alpha)
    return ImageClip(numpy.array(image))


def add_captions_with_padding(
    video_file: str,
    output_file: str,
    segments: list[dict],
    caption_type: str,
    font_type: str,
    font_size: int,
    font_color: str,
    stroke_color: str,
    stroke_width: int,
    highlight_current_word: bool,
    word_highlight_color: str,
    line_count: int,
    padding: int,
    top_padding: int,
    bottom_padding: int,
    shadow_strength: float,
    shadow_blur: float,
) -> None:
    start_time = time.time()
    font = resolve_font(font_type)
    text_clip_padding = max(stroke_width * 4 + 12, int(font_size * TEXT_RENDER_PADDING_RATIO))

    print("Generating video elements...")
    video = VideoFileClip(video_file)
    text_bbox_width = video.w - padding * 2
    clips = [video]

    captions = captacity.segment_parser.parse(
        segments=segments,
        fit_function=captacity.fits_frame(
            line_count,
            font,
            font_size,
            stroke_width,
            text_bbox_width,
        ),
    )

    for caption in captions:
        captions_to_draw = []
        if highlight_current_word:
            for i, word in enumerate(caption["words"]):
                if i + 1 < len(caption["words"]):
                    end = caption["words"][i + 1]["start"]
                else:
                    end = word["end"]

                captions_to_draw.append(
                    {
                        "text": caption["text"],
                        "start": word["start"],
                        "end": end,
                    }
                )
        else:
            captions_to_draw.append(caption)

        for current_index, caption_to_draw in enumerate(captions_to_draw):
            line_data = captacity.calculate_lines(
                caption_to_draw["text"],
                font,
                font_size,
                stroke_width,
                text_bbox_width,
            )
            text_y_offset = get_caption_y(
                video.h,
                line_data["height"],
                top_padding,
                bottom_padding,
            )

            index = 0
            for line in line_data["lines"]:
                pos = ("center", text_y_offset - text_clip_padding)
                line_words = line["text"].split()
                first_word_index = index

                shadow_left = shadow_strength
                while shadow_left >= 1:
                    shadow_left -= 1
                    shadow = create_line_shadow_clip(
                        line_words,
                        caption_type,
                        font_size,
                        font,
                        stroke_width,
                        text_clip_padding,
                        shadow_blur,
                        opacity=1,
                    )
                    shadow = shadow.set_start(caption_to_draw["start"])
                    shadow = shadow.set_duration(
                        caption_to_draw["end"] - caption_to_draw["start"]
                    )
                    shadow = shadow.set_position(pos)
                    clips.append(shadow)

                if shadow_left > 0:
                    shadow = create_line_shadow_clip(
                        line_words,
                        caption_type,
                        font_size,
                        font,
                        stroke_width,
                        text_clip_padding,
                        shadow_blur,
                        opacity=shadow_left,
                    )
                    shadow = shadow.set_start(caption_to_draw["start"])
                    shadow = shadow.set_duration(
                        caption_to_draw["end"] - caption_to_draw["start"]
                    )
                    shadow = shadow.set_position(pos)
                    clips.append(shadow)

                text = create_line_clip(
                    words=line_words,
                    current_index=current_index if highlight_current_word else None,
                    first_word_index=first_word_index,
                    caption_type=caption_type,
                    font_size=font_size,
                    font_color=font_color,
                    word_highlight_color=word_highlight_color,
                    font=font,
                    stroke_color=stroke_color,
                    stroke_width=stroke_width,
                    text_padding=text_clip_padding,
                )
                text = text.set_start(caption_to_draw["start"])
                text = text.set_duration(caption_to_draw["end"] - caption_to_draw["start"])
                text = text.set_position(pos)
                clips.append(text)

                index += len(line_words)
                line_visual_height = max(line["height"], text.size[1] - text_clip_padding * 2)
                text_y_offset += line_visual_height

    generation_time = time.time() - start_time
    print(f"Generated in {generation_time // 60:02.0f}:{generation_time % 60:02.0f} ({len(clips)} clips)")
    print("Rendering video...")

    video_with_text = CompositeVideoClip(clips)
    video_with_text.write_videofile(
        filename=output_file,
        codec="libx264",
        fps=video.fps,
        logger="bar",
    )

    total_time = time.time() - start_time
    render_time = total_time - generation_time
    print(f"Generated in {generation_time // 60:02.0f}:{generation_time % 60:02.0f}")
    print(f"Rendered in {render_time // 60:02.0f}:{render_time % 60:02.0f}")
    print(f"Done in {total_time // 60:02.0f}:{total_time % 60:02.0f}")


def transcribe_for_editing(video_path: str | None, progress=gr.Progress(track_tqdm=True)):
    progress(0.05, desc="Validating upload")
    yield progress_status(5, "Validating upload..."), "", None

    try:
        ffmpeg_warning = check_ffmpeg()
        if ffmpeg_warning:
            yield ffmpeg_warning, "", None
            return

        source_path = validate_video(video_path)
        run_dir = APP_TEMP_ROOT / uuid.uuid4().hex
        run_dir.mkdir(parents=True, exist_ok=True)
        input_path = copy_upload(source_path, run_dir)

        progress(0.15, desc="Transcribing")
        yield progress_status(15, "Transcribing for editing..."), "", None
        segments = transcribe_video(str(input_path))

        progress(1.0, desc="Done")
        yield progress_status(100, "Transcript ready. Edit the text, then generate captions."), segments_to_text(segments), segments
    except gr.Error as exc:
        yield str(exc), "", None
    except Exception as exc:
        traceback.print_exc()
        yield (
            "Transcription failed. Check that openai-whisper is installed, "
            f"the video is readable, and ffmpeg is available. Details: {exc}"
        ), "", None


def generate_captions(
    video_path: str | None,
    transcript_text: str | None,
    transcript_segments: list[dict] | None,
    caption_type: str,
    font_type: str,
    font_size: int | float,
    font_color: str,
    stroke_color: str,
    stroke_width: int | float,
    highlight_current_word: bool,
    word_highlight_color: str,
    line_count: int,
    padding: int | float,
    top_padding: int | float,
    bottom_padding: int | float,
    shadow_strength: int | float,
    shadow_blur: int | float,
    progress=gr.Progress(track_tqdm=True),
):
    progress(0.05, desc="Validating upload")
    yield progress_status(5, "Validating upload..."), None

    try:
        ffmpeg_warning = check_ffmpeg()
        if ffmpeg_warning:
            yield ffmpeg_warning, None
            return

        source_path = validate_video(video_path)
        run_dir = APP_TEMP_ROOT / uuid.uuid4().hex
        run_dir.mkdir(parents=True, exist_ok=True)

        input_path = copy_upload(source_path, run_dir)
        rendered_path = run_dir / RENDER_OUTPUT_NAME
        output_path = run_dir / DEFAULT_OUTPUT_NAME

        progress(0.15, desc="Transcribing")
        if transcript_segments and transcript_text and transcript_text.strip():
            yield progress_status(15, "Applying edited transcript..."), None
            segments = edited_text_to_segments(transcript_text, transcript_segments)
        else:
            yield progress_status(15, "Transcribing..."), None
            segments = transcribe_video(str(input_path))

        progress(0.35, desc="Rendering captions")
        yield progress_status(35, "Rendering captions..."), None

        add_captions_with_padding(
            video_file=str(input_path),
            output_file=str(rendered_path),
            segments=segments,
            caption_type=caption_type,
            font_type=font_type,
            font_size=int(font_size),
            font_color=font_color,
            stroke_color=stroke_color,
            stroke_width=int(stroke_width),
            highlight_current_word=bool(highlight_current_word),
            word_highlight_color=word_highlight_color,
            line_count=int(line_count),
            padding=int(padding),
            top_padding=int(top_padding),
            bottom_padding=int(bottom_padding),
            shadow_strength=float(shadow_strength),
            shadow_blur=float(shadow_blur),
        )

        if not rendered_path.exists():
            yield "Rendering finished, but no captioned video was created.", None
            return

        progress(0.90, desc="Restoring original audio")
        yield progress_status(90, "Restoring original audio..."), None
        restore_original_audio(rendered_path, input_path, output_path)

        if not output_path.exists():
            yield "Rendering finished, but no output video was created.", None
            return

        progress(1.0, desc="Done")
        yield progress_status(100, "Done."), str(output_path)
    except gr.Error as exc:
        yield str(exc), None
    except Exception as exc:
        traceback.print_exc()
        yield (
            "Caption generation failed. Check that openai-whisper is installed, "
            "the video is readable, and ffmpeg is available. "
            f"Details: {exc}"
        ), None


def build_app() -> gr.Blocks:
    ffmpeg_warning = check_ffmpeg()
    startup_message = (
        ffmpeg_warning
        if ffmpeg_warning
        else "Ready. Upload an MP4, MOV, or MKV video and generate local captions."
    )

    with gr.Blocks(title="Captacity Caption Studio") as app:
        gr.Markdown(startup_message)

        with gr.Row():
            with gr.Column():
                input_video = gr.Video(label="Video", sources=["upload"], format=None)
                transcribe_button = gr.Button("Transcribe for Editing")
                generate_button = gr.Button("Generate Captions", variant="primary")
                status_box = gr.Textbox(
                    label="Status",
                    value="Ready.",
                    interactive=False,
                )
                transcript_state = gr.State(value=None)
                transcript_box = gr.Textbox(
                    label="Editable Transcript",
                    lines=6,
                    placeholder="Click Transcribe for Editing, then fix any words here before generating.",
                )
            with gr.Column():
                output_video = gr.Video(label="Output Video")

        with gr.Accordion("Caption Style", open=True):
            with gr.Row():
                caption_type = gr.Dropdown(
                    label="Caption Type",
                    choices=CAPTION_TYPES,
                    value=CAPTION_TYPE_COLOR,
                )
                font_type = gr.Dropdown(
                    label="Font Family",
                    choices=list(FONT_CHOICES.keys()),
                    value="Coplette",
                )
                font_size = gr.Number(label="Font Size", value=130, precision=0)
                font_color = gr.ColorPicker(label="Font Color", value="white")
                stroke_color = gr.ColorPicker(label="Stroke Color", value="black")
                stroke_width = gr.Number(label="Stroke Width", value=3, precision=0)
            with gr.Row():
                highlight_current_word = gr.Checkbox(
                    label="Highlight Current Word",
                    value=True,
                )
                word_highlight_color = gr.ColorPicker(
                    label="Word Highlight Color",
                    value="#FB6487",
                )
                line_count = gr.Radio(label="Line Count", choices=[1, 2, 3], value=2)
                padding = gr.Number(label="Padding", value=64, precision=0)
                top_padding = gr.Number(label="Top Padding", value=500, precision=0)
                bottom_padding = gr.Number(label="Bottom Padding", value=0, precision=0)
            with gr.Row():
                shadow_strength = gr.Number(label="Shadow Strength", value=1.0)
                shadow_blur = gr.Number(label="Shadow Blur", value=0.1)
            caption_type_preview_box = gr.HTML(
                value=caption_type_preview(CAPTION_TYPE_COLOR, "Coplette", 130, "#FB6487"),
                label="Caption Type Preview",
            )

        preview_inputs = [caption_type, font_type, font_size, word_highlight_color]
        for preview_input in preview_inputs:
            preview_input.change(
                fn=caption_type_preview,
                inputs=preview_inputs,
                outputs=[caption_type_preview_box],
            )

        transcribe_button.click(
            fn=transcribe_for_editing,
            inputs=[input_video],
            outputs=[status_box, transcript_box, transcript_state],
        )

        generate_button.click(
            fn=generate_captions,
            inputs=[
                input_video,
                transcript_box,
                transcript_state,
                caption_type,
                font_type,
                font_size,
                font_color,
                stroke_color,
                stroke_width,
                highlight_current_word,
                word_highlight_color,
                line_count,
                padding,
                top_padding,
                bottom_padding,
                shadow_strength,
                shadow_blur,
            ],
            outputs=[status_box, output_video],
        )

    return app


if __name__ == "__main__":
    default_server_name = "0.0.0.0" if os.getenv("SPACE_ID") else "127.0.0.1"
    server_name = os.getenv("GRADIO_SERVER_NAME", default_server_name)
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    build_app().queue().launch(server_name=server_name, server_port=server_port)
