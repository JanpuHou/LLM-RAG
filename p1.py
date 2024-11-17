from PyPDF2 import PdfReader

reader = PdfReader("20230203_alphabet_10K.pdf")
number_of_pages = len(reader.pages)
page = reader.pages[28]
text = page.extract_text()
print(number_of_pages)
print(page)
print(text)