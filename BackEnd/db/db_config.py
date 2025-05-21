import pymysql
from pymysql.cursors import DictCursor

def get_connection():
    try:
        return pymysql.connect(
            host='localhost',
            user='root',
            password='1234',
            db='capstone',
            charset='utf8mb4',
            cursorclass=DictCursor,
            connect_timeout=5
        )
    except pymysql.MySQLError as e:
        print(f"데이터베이스 연결 오류: {e}")
        raise