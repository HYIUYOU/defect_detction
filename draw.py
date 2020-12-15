

from PIL import ImageDraw,Image
path = 'testImg/1.jpg'
image = Image.open(path) # 打开一张图片
draw = ImageDraw.Draw(image) # 在上面画画
draw.rectangle([250, 50, 950, 700],outline=(255)) # [左上角x，左上角y，右下角x，右下角y]，outline边框颜色
image.show()

