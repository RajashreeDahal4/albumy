# -*- coding: utf-8 -*-
"""
    :author: Grey Li (李辉)
    :url: http://greyli.com
    :copyright: © 2018 Grey Li <withlihui@gmail.com>
    :license: MIT, see LICENSE for more details.
"""
import os
import random

from PIL import Image
from faker import Faker
from flask import current_app
from sqlalchemy.exc import IntegrityError

from albumy.extensions import db
from albumy.models import User, Photo, Tag, Comment, Notification

from transformers import BlipProcessor, BlipForConditionalGeneration
import requests
import torch
import json
import piexif

fake = Faker()


def fake_admin():
    admin = User(name='Grey Li',
                 username='greyli',
                 email='admin@helloflask.com',
                 bio=fake.sentence(),
                 website='http://greyli.com',
                 confirmed=True)
    admin.set_password('helloflask')
    notification = Notification(message='Hello, welcome to Albumy.', receiver=admin)
    db.session.add(notification)
    db.session.add(admin)
    db.session.commit()


def fake_user(count=10):
    for i in range(count):
        user = User(name=fake.name(),
                    confirmed=True,
                    username=fake.user_name(),
                    bio=fake.sentence(),
                    location=fake.city(),
                    website=fake.url(),
                    member_since=fake.date_this_decade(),
                    email=fake.email())
        user.set_password('123456')
        db.session.add(user)
        try:
            db.session.commit()
        except IntegrityError:
            db.session.rollback()


def fake_follow(count=30):
    for i in range(count):
        user = User.query.get(random.randint(1, User.query.count()))
        user.follow(User.query.get(random.randint(1, User.query.count())))
    db.session.commit()


def fake_tag(count=20):
    for i in range(count):
        tag = Tag(name=fake.word())
        db.session.add(tag)
        try:
            db.session.commit()
        except IntegrityError:
            db.session.rollback()

def generate_alt_text(image,processor,model):
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    alternative_text = processor.decode(out[0], skip_special_tokens=True)
    return alternative_text


def detect_objects(image_path,model):
    """
    Detect objects in the image using YOLOv5 and return a list of detected object names.
    """
    results = model(image_path)  # Run detection
    detected_objects = results.pandas().xywh[0]['name'].tolist()  # Extract object names
    return detected_objects

def save_metadata(image_path, detected_objects):
    """
    Save detected objects into image metadata (Exif).
    """
    img = Image.open(image_path)

    # Convert detected objects list to JSON string
    metadata = {"detected_objects": detected_objects}
    metadata_str = json.dumps(metadata)

    # Load existing Exif data safely
    exif_data = img.info.get("exif", None)
    if exif_data:
        exif_dict = piexif.load(exif_data)
    else:
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "Interop": {}, "1st": {}, "thumbnail": None}

    # Store metadata in UserComment field
    exif_dict["Exif"][piexif.ExifIFD.UserComment] = metadata_str.encode("utf-8")

    # Save image with updated metadata
    exif_bytes = piexif.dump(exif_dict)
    img.save(image_path, "jpeg", exif=exif_bytes)
    print(f"Metadata saved: {metadata_str}")


def fake_photo(count=30):
    # photos
    upload_path = current_app.config['ALBUMY_UPLOAD_PATH']
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model2 = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    for i in range(count):
        print(i)

        filename = 'random_%d.jpg' % i
        r = lambda: random.randint(128, 255)
        img = Image.new(mode='RGB', size=(800, 800), color=(r(), r(), r()))
        img.save(os.path.join(upload_path, filename))
        alternative_text=generate_alt_text(img,processor,model)
        detected_objs=detect_objects(os.path.join(upload_path, filename),model2)
        save_metadata(os.path.join(upload_path, filename), detected_objs)

        photo = Photo(
            description=alternative_text,
            filename=filename,
            filename_m=filename,
            filename_s=filename,
            author=User.query.get(random.randint(1, User.query.count())),
            timestamp=fake.date_time_this_year(),
        )

        # tags
        if len(detected_objs)==0:
            for j in range(random.randint(1, 5)):
                tag = Tag.query.get(random.randint(1, Tag.query.count()))
                if tag not in photo.tags:
                    photo.tags.append(tag)

        db.session.add(photo)
    db.session.commit()


def fake_collect(count=50):
    for i in range(count):
        user = User.query.get(random.randint(1, User.query.count()))
        user.collect(Photo.query.get(random.randint(1, Photo.query.count())))
    db.session.commit()


def fake_comment(count=100):
    for i in range(count):
        comment = Comment(
            author=User.query.get(random.randint(1, User.query.count())),
            body=fake.sentence(),
            timestamp=fake.date_time_this_year(),
            photo=Photo.query.get(random.randint(1, Photo.query.count()))
        )
        db.session.add(comment)
    db.session.commit()
