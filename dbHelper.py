import sqlite3
conn = sqlite3.connect('universe_price.db', isolation_level=None)
cur = conn.cursor()

#cur.execute('''CREATE TABLE balance
#(code varchar(6) PRIMARY KEY,
#bid_price int(20) NOT NULL,
#quantity int(20) NOT NULL,
#created_at varchar(14) NOT NULL,
#will_clear_at varchar(14)
#)''')

###INSERT Test
#sql = "insert into balance(code, bid_price, quantity, created_at, will_clear_at) values(?, ?, ?, ?, ?)"
#cur.execute(sql, ('005930', 70000, 10, '20201222', 'today'))
#print(cur.rowcount)

###SELECT Test
cur.execute("select * from balance")

row = cur.fetchone()
print(row)