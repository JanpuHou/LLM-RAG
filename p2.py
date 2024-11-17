from pdf2docx import Converter

def convert_pdf_to_docx(pdf_file, docx_file):

    # Create a Converter object
    cv = Converter(pdf_file)

    # Convert specified PDF page to docx 
    cv.convert(docx_file, start=0, end=None)
    cv.close()

# Convert a PDF to a Docx file
convert_pdf_to_docx("20230203_alphabet_10K.pdf", "Output.docx")