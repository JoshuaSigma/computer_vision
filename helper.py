from io import BytesIO
from PIL import Image
import requests

def load_img(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img