from fastapi import FastAPI, File, UploadFile 
import tensorflow as tf
import json 
from model_definition import SegmentationModel 

app = FastAPI() 

model = SegmentationModel().model
model.load_weights('cancer_weights.h5') 

@app.post('/')
async def scoring_endpoint(data: UploadFile = File(...)): 
    image_bytes = await data.read()
    image = tf.io.decode_image(image_bytes) 
    yhat = model.predict(tf.expand_dims(image, axis=0))
    return {"prediction": json.dumps(yhat.tolist())}
