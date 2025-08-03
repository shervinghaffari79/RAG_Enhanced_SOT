# Sketch of Thought (SoT) Repository 🚀

## Overview 📖
The Sketch of Thought (SoT) project is a machine learning repository designed to leverage advanced natural language processing and deep learning techniques. Built with PyTorch and the Hugging Face Transformers library, this project aims to provide a robust framework for experimenting with transformer-based models, potentially enhanced with Retrieval-Augmented Generation (RAG) techniques.  
This repository includes a Jupyter notebook (<code>RAG_Enhanced_SoT.ipynb</code>) that sets up the environment, installs dependencies, and prepares the groundwork for running experiments with the SoT framework.

## Table of Contents 📑
- [Installation](#installation) 🔧
- [Dependencies](#dependencies) 📦
- [Setup Instructions](#setup-instructions) 🛠️
- [Usage](#usage) ▶️
- [Contributing](#contributing) 🤝
- [License](#license) 📄
- [Contact](#contact) 📧

## Installation 🔧
To get started with the SoT repository, follow these steps to set up the environment and install the necessary dependencies.

### Prerequisites ✅
- Python 3.10 or higher 🐍
- Git 📂
- A compatible environment (e.g., Google Colab, local Jupyter Notebook, or a cloud-based IDE) ☁️
- Access to a GPU (recommended for faster model training and inference) ⚡

### Clone the Repository 📥
Clone the repository to your local machine or cloud environment:
```bash
git clone https://github.com/SimonAytes/SoT.git
cd SoT
```

## Dependencies 📦
The project relies on the following key dependencies, as specified in the notebook:

- PyTorch (<code>torch==2.0.1</code>): Deep learning framework for model training and inference. 🔥
- Transformers (<code>transformers==4.30.0</code>): Hugging Face library for transformer models. 🤗
- Tokenizers (<code>tokenizers==0.13.3</code>): For text tokenization compatible with Transformers. ✂️
- NumPy (<code>numpy==1.26.4</code>): For numerical computations. 🔢
- Loguru (<code>loguru==0.7.3</code>): For logging and debugging. 📝

Additional dependencies include CUDA libraries for GPU support and other utilities listed in <code>requirements.txt</code>.

## Setup Instructions 🛠️

1. **Navigate to the Project Directory:**
   ```bash
   cd SoT
   ```

2. **Create a Virtual Environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   Install the required packages using the provided <code>requirements.txt</code>:
   ```bash
   pip install -r requirements.txt
   ```

   Alternatively, if you are using the provided Jupyter notebook, run the following cell to install dependencies:
   ```
   !pip install -r requirements.txt
   ```

4. **Install the SoT Package:**
   The repository includes a custom package (<code>sketch_of_thought</code>). Install it in editable mode:
   ```bash
   pip install -e .
   ```

5. **Verify Installation:**
   Ensure all dependencies are installed correctly by running the Jupyter notebook (<code>RAG_Enhanced_SoT.ipynb</code>) or by importing the required libraries in a Python script:
   ```python
   import torch
   import transformers
   print(torch.__version__)  # Should print 2.0.1
   print(transformers.__version__)  # Should print 4.30.0
   ```

## Usage ▶️
The <code>RAG_Enhanced_SoT.ipynb</code> notebook provides a starting point for working with the SoT framework. Follow these steps to use it:

1. **Open the Notebook:**
   Launch Jupyter Notebook or open the notebook in a compatible environment like Google Colab:
   ```bash
   jupyter notebook RAG_Enhanced_SoT.ipynb
   ```

2. **Run the Setup Cells:**
   Execute the cells in the notebook to:
   - Clone the repository (if not already done). 📥
   - Install dependencies. 📦
   - Download model weights (e.g., transformer model shards). ⬇️

3. **Experiment with the Framework:**
   The notebook likely includes cells for loading models, processing data, and running experiments. Modify these cells to suit your use case, such as fine-tuning a transformer model or implementing RAG-based inference. 🧪

### Example Workflow 💡
**Example: Loading a transformer model (modify as per your needs)**
```python
from transformers import AutoModel, AutoTokenizer

model_name = "your-model-name"  # Replace with the specific model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Tokenize input
inputs = tokenizer("Your input text here", return_tensors="pt")

# Run inference
outputs = model(**inputs)
```

## Contributing 🤝
We welcome contributions to improve the SoT project! To contribute:

1. Fork the repository. 🍴
2. Create a new branch (<code>git checkout -b feature/your-feature</code>). 🌿
3. Make your changes and commit (<code>git commit -m "Add your feature"</code>). 💾
4. Push to the branch (<code>git push origin feature/your-feature</code>). 📤
5. Open a Pull Request. 🔄

Please ensure your code follows the project's coding standards and includes appropriate tests. 🧑‍🔬

## License 📄
This project is licensed under the MIT License. See the <a href="LICENSE">LICENSE</a> file for details. ⚖️

## Contact 📧
For questions or support, please open an issue on the <a href="https://github.com/SimonAytes/SoT/issues">GitHub Issues page</a> or contact the maintainer at <a href="mailto:your-email@example.com">your-email@example.com</a>. 💬

Happy experimenting with Sketch of Thought! 🎉
