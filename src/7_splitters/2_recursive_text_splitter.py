# ["\n\n", "\n", " ", ""]  tries to split by order till chunk fits given limit
from langchain_text_splitters import RecursiveCharacterTextSplitter

sample_text = """
Artificial intelligence has evolved dramatically over the past decade, shifting from rigid rule-based systems to highly adaptable probabilistic models. Machine learning, a core subset of AI, enables computers to learn from data without being explicitly programmed for every single edge case.

Supervised learning remains the most commercially successful paradigm. In this approach, models are trained on labeled datasets—meaning every input comes with a corresponding correct output. Common applications include spam detection, image recognition, and predictive maintenance in industrial machinery.

Unsupervised learning, on the other hand, deals with unlabeled data. The algorithm must discover hidden structures, clusters, or patterns on its own. Clustering algorithms like K-Means and dimensionality reduction techniques like Principal Component Analysis (PCA) are prime examples of this methodology.

Key milestones in modern machine learning:
1. The resurgence of deep learning and multi-layer neural networks.
2. The introduction of the Transformer architecture, revolutionizing natural language processing.
3. The rise of self-supervised learning, enabling models to train on massive unannotated corpora.

Ultimately, understanding how text splitters handle these varying structures ensures that your retrieval-augmented generation (RAG) pipeline feeds clean, semantically coherent chunks into your vector database.
"""

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
result = splitter.split_text(sample_text)
print(result)
