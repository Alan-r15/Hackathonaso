from tensorflow import keras
from PIL import Image
import numpy as np

class mlmodel():
	def __init__(self):
		self.team = "Los voladores de Papantla"
		self.name = "Hentai Lover De Durango"
		self.model = keras.models.load_model("Aver128.model",compile=False)
		self.labels = ["Apple","Banana","Orange","Mixed"]

	def predict(self,path = "Appleasa.jpg"):
		top,right,bottom,left = (0,0,0,0)
		coords = (top,right,bottom,left)

		try:
			im = Image.open(path).convert('L')
			im = im.resize((128,128))
			np_im = np.array(im)  # convert to array
			np_im = np_im.reshape(-1,128,128,1)

			prediction = self.model.predict(np_im)
			predicted_fruits  = []
			for i in range(0,4):
				predicted_fruits.append((self.labels[i],prediction[0,i],coords))
			return predicted_fruits
		except:
			return None
