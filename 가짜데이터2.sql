use capstone;
INSERT INTO videos (video_id, title, location, recorded_date, damage_image_count) VALUES (11, '영상_1', '동작구', '2025-04-07', 13);
INSERT INTO videos (video_id, title, location, recorded_date, damage_image_count) VALUES (12, '영상_2', '도봉구', '2025-05-23', 12);
INSERT INTO videos (video_id, title, location, recorded_date, damage_image_count) VALUES (13, '영상_3', '은평구', '2025-06-25', 9);
INSERT INTO videos (video_id, title, location, recorded_date, damage_image_count) VALUES (14, '영상_4', '동작구', '2025-07-15', 14);
INSERT INTO videos (video_id, title, location, recorded_date, damage_image_count) VALUES (15, '영상_5', '금천구', '2025-08-09', 9);
INSERT INTO videos (video_id, title, location, recorded_date, damage_image_count) VALUES (16, '영상_6', '성북구', '2025-09-19', 14);
INSERT INTO videos (video_id, title, location, recorded_date, damage_image_count) VALUES (17, '영상_7', '구로구', '2025-010-18', 10);
INSERT INTO videos (video_id, title, location, recorded_date, damage_image_count) VALUES (18, '영상_8', '중랑구', '2025-11-09', 12);
INSERT INTO videos (video_id, title, location, recorded_date, damage_image_count) VALUES (19, '영상_9', '중구', '2025-12-19', 10);
INSERT INTO videos (video_id, title, location, recorded_date, damage_image_count) VALUES (20, '영상_10', '광진구', '2025-02-15', 9);

INSERT INTO damage_images (video_id, image_title, damage_type, timeline, image_path) VALUES (11, 'image_0001', '토사퇴적', '0:08:37', '/images/video_1/frame_0001.jpg');
INSERT INTO damage_images (video_id, image_title, damage_type, timeline, image_path) VALUES (11, 'image_0002', '균열', '0:03:33', '/images/video_1/frame_0002.jpg');
INSERT INTO damage_images (video_id, image_title, damage_type, timeline, image_path) VALUES (11, 'image_0003', '파손', '0:07:47', '/images/video_1/frame_0003.jpg');
INSERT INTO damage_images (video_id, image_title, damage_type, timeline, image_path) VALUES (11, 'image_0004', '표면손상', '0:07:11', '/images/video_1/frame_0004.jpg');
INSERT INTO damage_images (video_id, image_title, damage_type, timeline, image_path) VALUES (11, 'image_0005', '토사퇴적', '0:00:27', '/images/video_1/frame_0005.jpg');
INSERT INTO damage_images (video_id, image_title, damage_type, timeline, image_path) VALUES (11, 'image_0006', '기타', '0:00:19', '/images/video_1/frame_0006.jpg');
INSERT INTO damage_images (video_id, image_title, damage_type, timeline, image_path) VALUES (11, 'image_0007', '기타', '0:04:38', '/images/video_1/frame_0007.jpg');
INSERT INTO damage_images (video_id, image_title, damage_type, timeline, image_path) VALUES (11, 'image_0008', '연결관돌출', '0:01:28', '/images/video_1/frame_0008.jpg');
INSERT INTO damage_images (video_id, image_title, damage_type, timeline, image_path) VALUES (11, 'image_0009', '연결관돌출', '0:04:20', '/images/video_1/frame_0009.jpg');
INSERT INTO damage_images (video_id, image_title, damage_type, timeline, image_path) VALUES (11, 'image_0010', '파손', '0:01:24', '/images/video_1/frame_0010.jpg');

INSERT INTO damage_images (video_id, image_title, damage_type, timeline, image_path) VALUES (12, 'image_0011', '이음부단차', '0:00:33', '/images/video_2/frame_0011.jpg');
INSERT INTO damage_images (video_id, image_title, damage_type, timeline, image_path) VALUES (12, 'image_0012', '이음부단차', '0:02:15', '/images/video_2/frame_0012.jpg');
INSERT INTO damage_images (video_id, image_title, damage_type, timeline, image_path) VALUES (12, 'image_0013', '연결관돌출', '0:07:50', '/images/video_2/frame_0013.jpg');
INSERT INTO damage_images (video_id, image_title, damage_type, timeline, image_path) VALUES (12, 'image_0014', '기타', '0:08:28', '/images/video_2/frame_0014.jpg');
INSERT INTO damage_images (video_id, image_title, damage_type, timeline, image_path) VALUES (12, 'image_0015', '이음부손상', '0:01:43', '/images/video_2/frame_0015.jpg');
INSERT INTO damage_images (video_id, image_title, damage_type, timeline, image_path) VALUES (12, 'image_0016', '기타', '0:03:40', '/images/video_2/frame_0016.jpg');
INSERT INTO damage_images (video_id, image_title, damage_type, timeline, image_path) VALUES (12, 'image_0017', '기타', '0:08:27', '/images/video_2/frame_0017.jpg');
INSERT INTO damage_images (video_id, image_title, damage_type, timeline, image_path) VALUES (12, 'image_0018', '토사퇴적', '0:08:08', '/images/video_2/frame_0018.jpg');
INSERT INTO damage_images (video_id, image_title, damage_type, timeline, image_path) VALUES (12, 'image_0019', '이음부손상', '0:06:01', '/images/video_2/frame_0019.jpg');
INSERT INTO damage_images (video_id, image_title, damage_type, timeline, image_path) VALUES (12, 'image_0020', '기타', '0:06:42', '/images/video_2/frame_0020.jpg');

INSERT INTO damage_images (video_id, image_title, damage_type, timeline, image_path) VALUES (13, 'image_0021', '이음부단차', '0:05:07', '/images/video_3/frame_0021.jpg');
INSERT INTO damage_images (video_id, image_title, damage_type, timeline, image_path) VALUES (13, 'image_0022', '이음부단차', '0:03:21', '/images/video_3/frame_0022.jpg');
INSERT INTO damage_images (video_id, image_title, damage_type, timeline, image_path) VALUES (13, 'image_0023', '이음부손상', '0:07:14', '/images/video_3/frame_0023.jpg');
INSERT INTO damage_images (video_id, image_title, damage_type, timeline, image_path) VALUES (13, 'image_0024', '이음부손상', '0:06:23', '/images/video_3/frame_0024.jpg');
INSERT INTO damage_images (video_id, image_title, damage_type, timeline, image_path) VALUES (13, 'image_0025', '토사퇴적', '0:01:25', '/images/video_3/frame_0025.jpg');
INSERT INTO damage_images (video_id, image_title, damage_type, timeline, image_path) VALUES (13, 'image_0026', '균열', '0:08:42', '/images/video_3/frame_0026.jpg');
INSERT INTO damage_images (video_id, image_title, damage_type, timeline, image_path) VALUES (13, 'image_0027', '파손', '0:00:51', '/images/video_3/frame_0027.jpg');
INSERT INTO damage_images (video_id, image_title, damage_type, timeline, image_path) VALUES (13, 'image_0028', '연결관돌출', '0:08:10', '/images/video_3/frame_0028.jpg');
INSERT INTO damage_images (video_id, image_title, damage_type, timeline, image_path) VALUES (13, 'image_0029', '기타', '0:08:41', '/images/video_3/frame_0029.jpg');
INSERT INTO damage_images (video_id, image_title, damage_type, timeline, image_path) VALUES (13, 'image_0030', '이음부단차', '0:06:58', '/images/video_3/frame_0030.jpg');

INSERT INTO damage_images (video_id, image_title, damage_type, timeline, image_path)
VALUES (14, 'image_0031', '이음부손상', '0:01:47', '/images/video_4/frame_0031.jpg');

INSERT INTO damage_images (video_id, image_title, damage_type, timeline, image_path)
VALUES (14, 'image_0032', '연결관돌출', '0:03:21', '/images/video_4/frame_0032.jpg');

INSERT INTO damage_images (video_id, image_title, damage_type, timeline, image_path)
VALUES (14, 'image_0033', '토사퇴적', '0:05:02', '/images/video_4/frame_0033.jpg');

INSERT INTO damage_images (video_id, image_title, damage_type, timeline, image_path)
VALUES (14, 'image_0034', '균열', '0:02:15', '/images/video_4/frame_0034.jpg');

INSERT INTO damage_images (video_id, image_title, damage_type, timeline, image_path)
VALUES (14, 'image_0035', '기타', '0:08:03', '/images/video_4/frame_0035.jpg');

INSERT INTO damage_images (video_id, image_title, damage_type, timeline, image_path)
VALUES (15, 'image_0041', '이음부단차', '0:04:40', '/images/video_5/frame_0041.jpg');

INSERT INTO damage_images (video_id, image_title, damage_type, timeline, image_path)
VALUES (15, 'image_0042', '파손', '0:01:32', '/images/video_5/frame_0042.jpg');

INSERT INTO damage_images (video_id, image_title, damage_type, timeline, image_path)
VALUES (15, 'image_0043', '연결관돌출', '0:07:14', '/images/video_5/frame_0043.jpg');

INSERT INTO damage_images (video_id, image_title, damage_type, timeline, image_path)
VALUES (15, 'image_0044', '표면손상', '0:03:25', '/images/video_5/frame_0044.jpg');

INSERT INTO damage_images (video_id, image_title, damage_type, timeline, image_path)
VALUES (15, 'image_0045', '기타', '0:06:17', '/images/video_5/frame_0045.jpg');

-- video_id = 16
INSERT INTO damage_images (video_id, image_title, damage_type, timeline, image_path) VALUES
(16, 'image_0046', '이음부단차', '0:02:10', '/images/video_6/frame_0046.jpg'),
(16, 'image_0047', '토사퇴적', '0:03:55', '/images/video_6/frame_0047.jpg'),
(16, 'image_0048', '파손', '0:01:17', '/images/video_6/frame_0048.jpg'),
(16, 'image_0049', '표면손상', '0:06:05', '/images/video_6/frame_0049.jpg'),
(16, 'image_0050', '연결관돌출', '0:04:26', '/images/video_6/frame_0050.jpg');

-- video_id = 17
INSERT INTO damage_images (video_id, image_title, damage_type, timeline, image_path) VALUES
(17, 'image_0051', '기타', '0:00:45', '/images/video_7/frame_0051.jpg'),
(17, 'image_0052', '균열', '0:03:30', '/images/video_7/frame_0052.jpg'),
(17, 'image_0053', '토사퇴적', '0:05:20', '/images/video_7/frame_0053.jpg'),
(17, 'image_0054', '이음부손상', '0:06:40', '/images/video_7/frame_0054.jpg'),
(17, 'image_0055', '관벽박리', '0:02:15', '/images/video_7/frame_0055.jpg');

-- video_id = 18
INSERT INTO damage_images (video_id, image_title, damage_type, timeline, image_path) VALUES
(18, 'image_0056', '연결관돌출', '0:01:05', '/images/video_8/frame_0056.jpg'),
(18, 'image_0057', '기타', '0:02:50', '/images/video_8/frame_0057.jpg'),
(18, 'image_0058', '균열', '0:04:30', '/images/video_8/frame_0058.jpg'),
(18, 'image_0059', '표면손상', '0:07:10', '/images/video_8/frame_0059.jpg'),
(18, 'image_0060', '이음부단차', '0:03:45', '/images/video_8/frame_0060.jpg');

-- video_id = 19
INSERT INTO damage_images (video_id, image_title, damage_type, timeline, image_path) VALUES
(19, 'image_0061', '파손', '0:02:22', '/images/video_9/frame_0061.jpg'),
(19, 'image_0062', '이음부손상', '0:01:35', '/images/video_9/frame_0062.jpg'),
(19, 'image_0063', '토사퇴적', '0:05:44', '/images/video_9/frame_0063.jpg'),
(19, 'image_0064', '연결관돌출', '0:04:11', '/images/video_9/frame_0064.jpg'),
(19, 'image_0065', '기타', '0:03:03', '/images/video_9/frame_0065.jpg');

-- video_id = 20
INSERT INTO damage_images (video_id, image_title, damage_type, timeline, image_path) VALUES
(20, 'image_0066', '균열', '0:00:42', '/images/video_10/frame_0066.jpg'),
(20, 'image_0067', '이음부단차', '0:01:50', '/images/video_10/frame_0067.jpg'),
(20, 'image_0068', '관벽박리', '0:04:18', '/images/video_10/frame_0068.jpg'),
(20, 'image_0069', '파손', '0:06:23', '/images/video_10/frame_0069.jpg'),
(20, 'image_0070', '기타', '0:03:40', '/images/video_10/frame_0070.jpg');

