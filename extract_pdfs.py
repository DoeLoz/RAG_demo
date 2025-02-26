import os
import json
import pdfplumber
import re

# 预处理文本
def clean_text(text):
    text = re.sub(r'\s*(第.*页|page\s*\d+)\s*', '', text)
    text = re.sub(r'\s*（.*?）\s*', '', text)
    text = re.sub(r'[\n\s]+', ' ', text)
    return text

# 切分文本块
def split_text_with_overlap(text, chunk_size=189, overlap_size=52):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap_size):
        chunks.append(text[i:i + chunk_size])
    return chunks

# 读取 PDF 并提取文本
def extract_text_from_pdfs(pdf_folder, output_path):
    pdf_content = []
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    total_pdfs = len(pdf_files)

    print(f"开始处理 {total_pdfs} 个 PDF 文件...\n")

    for pdf_idx, filename in enumerate(pdf_files):
        print(f"正在处理第 {pdf_idx + 1}/{total_pdfs} 个文件: {filename}")
        pdf_path = os.path.join(pdf_folder, filename)

        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            for page_idx, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    cleaned_text = clean_text(text)
                    chunked_texts = split_text_with_overlap(cleaned_text)
                    for chunk in chunked_texts:
                        pdf_content.append({'file': filename, 'page': page_idx + 1, 'content': chunk})

            print(f"文件 {filename} 处理完毕，共 {total_pages} 页。")

    print(f"\nPDF 文件处理完成，共提取了 {len(pdf_content)} 个文本块。\n")

    # 保存到 JSON 文件
    with open(output_path, 'w', encoding='utf8') as file:
        json.dump(pdf_content, file, ensure_ascii=False, indent=4)

    print(f"数据已保存至 {output_path}")

if __name__ == "__main__":
    pdf_folder = "./financebench/pdfs"  # PDF 文件夹路径
    output_path = "./financebench/processed_pdfs.json"  # 处理后数据的存储路径
    extract_text_from_pdfs(pdf_folder, output_path)
