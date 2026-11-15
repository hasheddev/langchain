from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language,
)

code_string = """
class DataProcessor:
    def __init__(self, filepath: str):
        self.filepath = filepath
        
    def load_data(self):
        print(f"Loading data from {self.filepath}")
        return {"status": "success", "rows": 100}

    def process(self, data):
        processed = [row for row in range(data["rows"])]
        return processed

def main():
    processor = DataProcessor("data.csv")
    data = processor.load_data()
    print("Data loaded successfully.")

if __name__ == "__main__":
    main()
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    Language.PYTHON, chunk_size=150, chunk_overlap=20
)

result = splitter.split_text(code_string)

print(result)
