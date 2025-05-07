# db/init_db.py
import pymysql

def initialize_database():
    connection = pymysql.connect(
        host='localhost',
        user='root',
        password='1234',
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

    try:
        with connection.cursor() as cursor:
            cursor.execute("CREATE DATABASE IF NOT EXISTS capstone")
            connection.commit()

        connection.select_db('capstone')
        with connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS videos (
                    video_id INT AUTO_INCREMENT PRIMARY KEY,
                    title VARCHAR(255) NOT NULL,
                    location VARCHAR(100),
                    recorded_date DATE,
                    damage_image_count INT DEFAULT 0
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS damage_images (
                    image_id INT AUTO_INCREMENT PRIMARY KEY,
                    video_id INT,
                    image_title VARCHAR(255),
                    damage_type VARCHAR(100),
                    timeline TIME,
                    image_path TEXT,
                    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
                )
            """)
            connection.commit()
    finally:
        connection.close()
