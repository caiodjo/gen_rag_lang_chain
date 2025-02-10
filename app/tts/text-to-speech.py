import os
from dotenv import load_dotenv
from langchain_community.tools import ElevenLabsText2SpeechTool
from elevenlabs.client import ElevenLabs
from elevenlabs import play

load_dotenv()

if __name__ == "__main__":
    print(os.environ["ELEVENLABS_API_KEY"])
    text_to_speak = "As contra indicações da nimesulida são várias"

    # tts = ElevenLabsText2SpeechTool()
    # print(tts.name)
    # speech_file = tts.run(text_to_speak)
    # tts.play(speech_file)

    client = ElevenLabs(
        api_key=os.environ["ELEVENLABS_API_KEY"],
    )
    audio = client.text_to_speech.convert(
        text=text_to_speak,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    play(audio)
