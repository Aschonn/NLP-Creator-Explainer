# be sure to follow the instruction on the readme or this will not work

import fasttext

nameofmodel = 'example'

predict_this_string = 'Enter in a string into this'

ourmodel = f'models/{nameofmodel}.bin'

model = fasttext.load_model(ourmodel)

print(model.predict(predict_this_string))