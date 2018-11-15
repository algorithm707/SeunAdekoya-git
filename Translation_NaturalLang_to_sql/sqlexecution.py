import sqlalchemy as s

sqlconn = 'root' + '' + '@localhost'

output_words, attentions = evaluate(encoder1, attn_decoder1, input_sentence)
query = ' '.join(output_words)

def executeQuery(query):
    db = s.create_engine(sqlconn)
    conn = db.connect()
    result = db.execute(query).fetchall()
    print (result)
