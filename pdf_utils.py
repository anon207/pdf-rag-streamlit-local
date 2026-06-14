from pypdf import PdfReader

def extract_pages(pdf_file) -> list[dict]: 
    """ 
    Returns: [{page_num: int, text: str}] page_num is 1-indexed for citations. 
    """ 
    reader = PdfReader(pdf_file) 
    pages = [] 
    for i, page in enumerate(reader.pages): 
        text = page.extract_text() or "" 
        text = " ".join(text.split()) 
        pages.append({"page_num": i + 1, "text": text}) 
    return pages