
import numpy as np

# Safely load the dictionary
data = np.load('label_map.npy', allow_pickle=True).item()
print(data)

'''
import pyttsx3

pred = 'I-am'

engine = pyttsx3.init()
engine.say(pred)
engine.runAndWait()

'''