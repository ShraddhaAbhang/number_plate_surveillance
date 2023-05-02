# main.py

from fastapi import FastAPI, status, UploadFile
from detection.detect_numberplate import detect

import uvicorn
app = FastAPI()

@app.post("/noplatedetection", status_code=status.HTTP_200_OK, tags=["Number Plate Surveilance"])
async def no_plate_api(image: UploadFile):
    numlist = await detect(image)

    return {"message": numlist}

if __name__ == "__main__":
    uvicorn.run("main:app", host = "127.0.0.1", port = 8444, reload = False, timeout_keep_alive=200)
