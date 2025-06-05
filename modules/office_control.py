import os
import time
import subprocess

try:
    import win32com.client
except ImportError:
    win32com = None

def open_word():
    try:
        subprocess.Popen(["start", "winword"], shell=True)
        return "Microsoft Word is opening."
    except Exception as e:
        return f"Failed to open Word: {e}"

def create_word_doc_with_text(text):
    if not win32com:
        return "win32com is not available. Please install pywin32 to use this feature."

    try:
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = True
        doc = word.Documents.Add()
        doc.Content.Text = text
        return "New Word document created with your text."
    except Exception as e:
        return f"Error creating Word document: {e}"

def open_excel():
    try:
        subprocess.Popen(["start", "excel"], shell=True)
        return "Microsoft Excel is opening."
    except Exception as e:
        return f"Failed to open Excel: {e}"
