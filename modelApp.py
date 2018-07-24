

import tkinter as tk
from PIL import Image, ImageTk
from keras.models import load_model
import keras
import numpy as np
from skimage import color
import imageio
from scipy import misc 

model = load_model('number-opt-7.h5')
model.compile(loss='binary_crossentropy',
              optimizer= keras.optimizers.RMSprop(lr=1e-4), 
              metrics=['accuracy'])


app = tk.Tk()

app.title = "Model Guesser App"
app.geometry("500x500")
app.configure(background="grey")

path = "test.png"
img = ImageTk.PhotoImage(Image.open(path))

panel = tk.Label(app, image = img)

#The Pack geometry manager packs widgets in rows or columns.
panel.pack(side = "top", fill = "both", expand = "yes")

def reload():
    img2 = ImageTk.PhotoImage(Image.open(path))
    panel.configure(image=img2)
    panel.image = img2


def classifyImage(): 
    imageInput = imageio.imread(l_text_1.get())
    if len(imageInput.shape) == 3:
        imageInput = color.rgb2gray(imageInput)
    imageInput = misc.imresize(imageInput, (200,200))
    imageInput = np.expand_dims(imageInput,axis = 0)
    imageInput = np.expand_dims(imageInput,axis = 3)
    modelPrediction = model.predict(imageInput)[0][0]
    predict_text.set(modelPrediction)
    img2 = ImageTk.PhotoImage(Image.open(l_text_1.get()))
    panel.configure(image=img2)
    panel.image = img2

l_text_1 = tk.StringVar()
l_text_1.set("test.png")
l_label = tk.Entry(app, textvariable=l_text_1,width = 15)
l_label.pack(side="left")

p = tk.Button(app, text="Classify", command=classifyImage)
p.pack(side = "bottom")

predict_text = tk.StringVar()
predictLabel = tk.Label(app, textvariable=predict_text,width = 5, height = 5)
predictLabel.pack(side = "top")


app.mainloop()
