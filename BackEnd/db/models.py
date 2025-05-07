# db/models.py
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Video(db.Model):
    __tablename__ = 'videos'
    video_id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    location = db.Column(db.String(100))
    recorded_date = db.Column(db.Date)
    damage_image_count = db.Column(db.Integer, default=0)

class DamageImage(db.Model):
    __tablename__ = 'damage_images'
    image_id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.Integer, db.ForeignKey('videos.video_id', ondelete='CASCADE'))
    image_title = db.Column(db.String(255))
    damage_type = db.Column(db.String(100))
    timeline = db.Column(db.Time)
    image_path = db.Column(db.Text)
