import edge_tts
import asyncio
import playsound
import os

DANGER_CLASSES = {"pavement_damage", "ramp"}

# 한국어 여성 목소리 옵션
# "ko-KR-SunHiNeural"   — 여성, 밝은 톤
# "ko-KR-IUNeural"      — 여성, 부드러운 톤
VOICE = "ko-KR-SunHiNeural"


async def _synthesize(message: str):
    communicate = edge_tts.Communicate(message, VOICE)
    await communicate.save("tts_temp.mp3")


def speak(message: str, last_message: str) -> str:
    if message == last_message:
        return last_message

    asyncio.run(_synthesize(message))
    playsound.playsound("tts_temp.mp3")
    try:
        os.remove("tts_temp.mp3")
    except Exception as e:
        print(f"삭제 오류: {e}")
    return message

def get_tts_message(class_name: str, avoid_direction: str) -> str | None:
    if class_name not in DANGER_CLASSES:
        return None

    messages = {
        "pavement_damage": f"전방에 노면 파손 감지. {avoid_direction}",
        "ramp": f"전방에 경사로 감지. {avoid_direction}",
    }
    return messages.get(class_name)


if __name__ == "__main__":
    last = ""
    last = speak("전방에 노면 파손 감지. 우측으로 우회하세요", last)
    print(f"1번: {last}")

    last = speak("전방에 노면 파손 감지. 우측으로 우회하세요", last)
    print(f"2번 스킵 확인: {last}")

    last = speak("전방에 경사로 감지. 좌측으로 우회하세요", last)
    print(f"3번: {last}")