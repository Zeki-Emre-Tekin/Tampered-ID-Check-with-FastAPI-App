from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import os
import cv2
import imutils
from skimage.metrics import structural_similarity
import subprocess

subprocess.check_call(['python3', '-m', 'pip', 'install', 'tensorflow'])

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure Jinja2Templates
templates = Jinja2Templates(directory="templates")

app.config = {
    'INITIAL_FILE_UPLOADS': 'static/uploads',
    'EXISTING_FILE': 'static/original',
    'GENERATED_FILE': 'static/generated'
}

if not os.path.exists(app.config['INITIAL_FILE_UPLOADS']):
    os.makedirs(app.config['INITIAL_FILE_UPLOADS'])

if not os.path.exists(app.config['EXISTING_FILE']):
    os.makedirs(app.config['EXISTING_FILE'])

if not os.path.exists(app.config['GENERATED_FILE']):
    os.makedirs(app.config['GENERATED_FILE'])

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def upload_image(request: Request, file_upload: UploadFile = File(...)):
    # Resize and save the uploaded image
    uploaded_image = Image.open(file_upload.file).resize((250, 160))
    uploaded_image.save(os.path.join(app.config['INITIAL_FILE_UPLOADS'], 'image.jpg'))

    # Resize and save the original image to ensure both uploaded and original matches in size
    original_image = Image.open(os.path.join(app.config['EXISTING_FILE'], 'image.jpg')).resize((250, 160))
    original_image.save(os.path.join(app.config['EXISTING_FILE'], 'image.jpg'))

    # Read uploaded and original image as array
    original_image = cv2.imread(os.path.join(app.config['EXISTING_FILE'], 'image.jpg'))
    uploaded_image = cv2.imread(os.path.join(app.config['INITIAL_FILE_UPLOADS'], 'image.jpg'))

    # Convert image into grayscale
    original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    uploaded_gray = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2GRAY)

    # Calculate structural similarity
    (score, diff) = structural_similarity(original_gray, uploaded_gray, full=True)
    diff = (diff * 255).astype("uint8")

    # Calculate threshold and contours
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Draw contours on image
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(uploaded_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Save all output images (if required)
    cv2.imwrite(os.path.join(app.config['GENERATED_FILE'], 'image_original.jpg'), original_image)
    cv2.imwrite(os.path.join(app.config['GENERATED_FILE'], 'image_uploaded.jpg'), uploaded_image)
    cv2.imwrite(os.path.join(app.config['GENERATED_FILE'], 'image_diff.jpg'), diff)
    cv2.imwrite(os.path.join(app.config['GENERATED_FILE'], 'image_thresh.jpg'), thresh)

    # Return result to the user
    return templates.TemplateResponse("index.html", {"request": request, "pred": f"{round(score * 100, 2)}% correct"})

