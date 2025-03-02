"""
This script is used to extract recipe information from YouTube videos. 
It uses the OpenAI API to generate a prompt for the user to extract recipe information from 
the subtitles of a YouTube video. 
The script then uses the YouTube Transcript API to extract the subtitles from the video
 and sends the subtitles to the OpenAI API to generate a prompt for 
 the user to extract recipe information from the subtitles. 
The script then converts the extracted recipe information into a Markdown recipe format
 and saves it to a file.
"""

from urllib.parse import parse_qs
from urllib.parse import urlparse
from openai import OpenAI
from yt_dlp import YoutubeDL
import timeout_decorator
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import find_dotenv, load_dotenv


DEFAULT_LANGUAGES = ["zh-TW", "zh-Hant", "zh", "zh-Hans", "ja", "en", "ko", "en-US"]


ALLOWED_NETLOCS = {
    "youtu.be",
    "m.youtube.com",
    "youtube.com",
    "www.youtube.com",
    "www.youtube-nocookie.com",
    "vid.plus",
}


class YoutubeLoader:
    def __init__(self, languages: list[str] | None = None) -> None:
        self.languages = languages or DEFAULT_LANGUAGES

    @timeout_decorator.timeout(20)
    def load(self, url: str) -> str:
        """
        Load youtube video subtitle
        Args:
            url (str): youtube video url
        Returns:
            str: video subtitle
        """
        video_id = parse_video_id(url)

        transcript_pieces: list[dict[str, str | float]] = (
            YouTubeTranscriptApi().get_transcript(video_id, self.languages)
        )

        lines = []
        for transcript_piece in transcript_pieces:
            text = str(transcript_piece.get("text", "")).strip()
            if text:
                lines.append(text)
        return "\n".join(lines)


def parse_video_id(url: str) -> str:
    """Parse a YouTube URL and return the video ID if valid, otherwise None."""
    parsed_url = urlparse(url)

    if parsed_url.scheme not in {"http", "https"}:
        raise f"unsupported URL scheme: {parsed_url.scheme}"

    if parsed_url.netloc not in ALLOWED_NETLOCS:
        raise f"unsupported URL netloc: {parsed_url.netloc}"

    path = parsed_url.path

    if path.endswith("/watch"):
        query = parsed_url.query
        parsed_query = parse_qs(query)
        if "v" in parsed_query:
            ids = parsed_query["v"]
            video_id = ids if isinstance(ids, str) else ids[0]
        else:
            raise f"no video found in URL: {url}"
    else:
        path = parsed_url.path.lstrip("/")
        video_id = path.split("/")[-1]

    if len(video_id) != 11:  # Video IDs are 11 characters long
        raise f"invalid video ID: {video_id}"

    return video_id


def get_youtube_videos(channel_url):
    """
    get youtube videos from channel

    Args:
        channel_url (str): channel url

    Returns:
        list: list of videos
    """
    ydl_opts = {
        "extract_flat": True,
        "quiet": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)

    video_list = [
        {"title": entry["title"], "url": entry["url"]}
        for entry in info.get("entries", [])
    ]
    return video_list


def main(channel_url):
    """
    get youtube videos subtitle and convert to markdown recipe
    """

    videos = get_youtube_videos(channel_url)
    load_dotenv(find_dotenv())

    client = OpenAI()
    yt_loader = YoutubeLoader()
    for video in videos:
        print(video["title"], video["url"])
        content = yt_loader.load(video["url"])

        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "請從以下字幕中抽取食譜資訊，請務必將所有內容翻譯成台灣繁體中文，且請按照以下 Markdown 格式輸出：\n\n"
                        "# 食譜標題\n\n"
                        "## 食材\n"
                        "- **食材名稱**: 數量 單位 (備註，如果有備註的話)\n\n"
                        "## 做法\n"
                        "完整的做法描述\n\n"
                        "請只輸出符合此格式的 Markdown 內容。"
                    ),
                },
                {"role": "user", "content": f"字幕：{content}"},
            ],
        )
        markdown_recipe = completion.choices[0].message.content
        with open(f"recipe/{video['title']}.md", "w", encoding="utf-8") as f:
            f.write(markdown_recipe)


if __name__ == "__main__":

    main("https://www.youtube.com/@kohkentetsukitchen/videos")
