import hashlib
import mysql.connector
conn = mysql.connector.connect(
   user='root', password='taylor@1989', host='localhost', database='weed'
)
def md5(input_string):
    md5_hash = hashlib.md5()
    md5_hash.update(input_string.encode('utf-8'))
    return md5_hash.hexdigest()
cursor = conn.cursor()
#Closing the connection
cursor.execute("DROP TABLE IF EXISTS tester")
#Closing the connection
sql ='''CREATE TABLE tester(
ID int NOT NULL AUTO_INCREMENT,
username VARCHAR(250) NOT NULL,
password VARCHAR(1000) not null,
email VARCHAR(250) not null,
primary key (id)
)'''
cursor.execute(sql)
cursor.execute("insert into tester(username,password,email) values (%s,%s,%s);",("tester",md5('tester123'),"tester@mail.com"))
conn.commit()