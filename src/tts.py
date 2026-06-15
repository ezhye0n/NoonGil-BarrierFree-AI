import edge_tts
import asyncio
import playsound
import os

DANGER_CLASSES = {
    "bench", "bicycle", "bollard", "clothing_bin",
    "cone", "electric_scooter", "fire_hydrant", "motorcycle",
    "pavement_damage", "ramp", "step", "street_light",
    "trash", "tree"
}

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
        "bench": f"전방에 벤치 감지. {avoid_direction}",
        "bicycle": f"전방에 자전거 감지. {avoid_direction}",
        "bollard": f"전방에 볼라드 감지. {avoid_direction}",
        "clothing_bin": f"전방에 의류수거함 감지. {avoid_direction}",
        "cone": f"전방에 라바콘 감지. {avoid_direction}",
        "electric_scooter": f"전방에 전동킥보드 감지. {avoid_direction}",
        "fire_hydrant": f"전방에 소화전 감지. {avoid_direction}",
        "motorcycle": f"전방에 오토바이 감지. {avoid_direction}",
        "pavement_damage": f"전방에 노면 파손 감지. {avoid_direction}",
        "ramp": f"전방에 경사로 감지. {avoid_direction}",
        "step": f"전방에 단차 감지. {avoid_direction}",
        "street_light": f"전방에 가로등 감지. {avoid_direction}",
        "trash": f"전방에 쓰레기통 감지. {avoid_direction}",
        "tree": f"전방에 가로수 감지. {avoid_direction}",
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