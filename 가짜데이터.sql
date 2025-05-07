create database capstone;
use capstone;
-- 1. 영상 테이블
CREATE TABLE videos (
    video_id INT AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    location VARCHAR(100),
    recorded_date DATE,
    damage_image_count INT DEFAULT 0
);

-- 2. 손상 이미지 테이블
CREATE TABLE damage_images (
    image_id INT AUTO_INCREMENT PRIMARY KEY,
    video_id INT,
    image_title VARCHAR(255),
    damage_type VARCHAR(100),
    timeline TIME,
    image_path TEXT,
    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
);

use capstone;
INSERT INTO videos (title, location, recorded_date, damage_image_count) VALUES
('강남구 하수관로 점검 1', '강남구', '2025-03-01', 3),
('서초구 하수관 점검 영상', '서초구', '2025-03-02', 2),
('마포구 CCTV 영상', '마포구', '2025-03-03', 1),
('송파구 관로 촬영', '송파구', '2025-03-04', 0),
('용산구 하수관 점검', '용산구', '2025-03-05', 2),
('은평구 관로 검사', '은평구', '2025-03-06', 1),
('노원구 하수도 영상', '노원구', '2025-03-07', 4),
('강서구 CCTV 검사', '강서구', '2025-03-08', 3),
('중구 하수관 촬영', '중구', '2025-03-09', 2),
('동작구 관로 분석', '동작구', '2025-03-10', 1);

INSERT INTO damage_images (video_id, image_title, damage_type, timeline, image_path) VALUES
(1, '강남_프레임_001', '균열', '00:01:05', '/images/강남/001.jpg'),
(1, '강남_프레임_002', '침하', '00:01:50', '/images/강남/002.jpg'),
(1, '강남_프레임_003', '이물질', '00:02:30', '/images/강남/003.jpg'),

(2, '서초_프레임_001', '침하', '00:00:45', '/images/서초/001.jpg'),
(2, '서초_프레임_002', '균열', '00:01:10', '/images/서초/002.jpg'),

(3, '마포_프레임_001', '균열', '00:02:00', '/images/마포/001.jpg'),

(5, '용산_프레임_001', '침하', '00:01:35', '/images/용산/001.jpg'),
(5, '용산_프레임_002', '관벽박리', '00:03:10', '/images/용산/002.jpg'),

(6, '은평_프레임_001', '이물질', '00:02:15', '/images/은평/001.jpg'),

(7, '노원_프레임_001', '균열', '00:00:30', '/images/노원/001.jpg'),
(7, '노원_프레임_002', '침하', '00:01:00', '/images/노원/002.jpg'),
(7, '노원_프레임_003', '관벽박리', '00:02:20', '/images/노원/003.jpg'),
(7, '노원_프레임_004', '균열', '00:02:50', '/images/노원/004.jpg'),

(8, '강서_프레임_001', '균열', '00:00:55', '/images/강서/001.jpg'),
(8, '강서_프레임_002', '침하', '00:01:45', '/images/강서/002.jpg'),
(8, '강서_프레임_003', '이물질', '00:03:05', '/images/강서/003.jpg'),

(9, '중구_프레임_001', '균열', '00:01:20', '/images/중구/001.jpg'),
(9, '중구_프레임_002', '관벽박리', '00:02:40', '/images/중구/002.jpg'),

(10, '동작_프레임_001', '이물질', '00:01:10', '/images/동작/001.jpg');

