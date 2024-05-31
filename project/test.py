from ultralytics import YOLO
import openai
import re

openai_api_key = ""
model = YOLO("yolov8n.pt")
results = model("/Users/tashapais/Downloads/yolo.jpeg", show= True)
print(results)

# results = "0: 384x640 1 person, 1 bottle, 1 refrigerator, 380.3ms"
# Modify the regular expression to match a word that follows a number and a space
# objects = re.findall(r'\d+ ([a-zA-Z]+)', results)
# print(objects)  # Output: ['person', 'bottle', 'refrigerator']

# Convert the list of objects to a set to remove duplicates, then convert it back to a list
# unique_objects = list(set(objects))
# print(unique_objects)

# response = openai.Completion.create(
#   engine="text-davinci-002",
#   prompt="Translate the following English text to French: '{Hello}'",
#   max_tokens=60
# )
#
# print(response.choices[0].text.strip())

# audio_file = open("/Users/tashapais/Downloads/audio.mp3", "rb")
# transcript = openai.Audio.transcribe("whisper-1", audio_file)
# print(transcript)
