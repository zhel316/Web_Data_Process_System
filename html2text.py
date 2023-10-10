import gzip
import re
from bs4 import BeautifulSoup

KEYNAME = "WARC-TREC-ID"
KEYHTML= "<!DOCTYPE html"

#according to the content "warc/1.0" , split the text then get the payload
def split_records(stream):
    payload = ''
    for line in stream:
        if line.strip() == "WARC/1.0":
            yield payload
            payload = ''
        else:
            payload += line
    yield payload

#find the text "WARC-TREC-ID" to get the key
def find_keys(payload):
    if payload == '':
        return

    key = None
    for line in payload.splitlines():
        if line.startswith(KEYNAME):
            key = line.split(': ')[1]
            break
    yield key


##transform the html resource to text
def html_to_text(record):
    html = ''
    flag = 0
    for line in record.splitlines():
        if line.startswith(KEYHTML):
            flag = 1
        if flag == 1 :
            html += line

    realHTML = html.replace('\n', '<br>')
    soup = BeautifulSoup(realHTML,features="html.parser")
    for script in soup(["script", "style","aside"]):
        script.extract()
    text = " ".join(re.split(r'[\n\t]+', soup.get_text()))
    text = re.sub(r"\s+", " ", text)
    text = re.sub("[^\u4e00-\u9fa5^\s\.\!\:\-\@\#\$\(\)\_\,\;\?^a-z^A-Z^0-9]","",text)
    return text


#decode the warc format file
def read_warc(file_path):
    INPUT = file_path
    with gzip.open(INPUT, 'rt', errors='ignore') as fo:
        file_content = fo.readlines()
        for record in split_records(file_content):
            for key in find_keys(record):
                if key:
                    text = html_to_text(record)
                    if text:
                        yield key, text

if __name__ == '__main__':
    for k,text in read_warc("data/sample.warc.gz"):
        print(k,text)