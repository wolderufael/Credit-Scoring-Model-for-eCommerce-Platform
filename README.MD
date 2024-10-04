
# Credit-Scoring-Model-for-eCommerce-Platform

This repository contains a credit scoring model designed for an eCommerce platform. The model leverages machine learning algorithms to assess the creditworthiness of potential customers, enabling the platform to make informed lending decisions.

## Project Structure
├── .github.<br>
│   └── workflow<br>
│       └── test.yaml<br>
├── .venv <br>
├── .vscode<br>
│   └── settings.json<br>
├── data<br>
│   ├── data.csv <br>
├── notebook<br>
│   ├── EDA.ipynb<br>
│
├── script
│   ├── EDA.py<br>
│   
├── .gitignore<br>
├──  README.MD<br>
├──  requirements.txt <br>

The following is an overview of the file structure for the project:
## Contents

### 1. `.github/workflow/test.yaml`
- Contains a GitHub Actions configuration for Continuous Integration (CI), specifically designed to run unit tests across multiple Python versions.

### 2. `data/`
- Contains various datasets used for NER tasks:
  - **data.csv**: Customer data.

### 3. `notebook/`
- Jupyter Notebooks:
  - **EDA.ipynb**: Steps for overview,summary,distribution,missing value check and outlier detection of data.

### 4. `script/`
- Python scripts:
  - **EDA.py**: Scripts for overview,summary,distribution,missing value check and outlier detection of data.

### 5. `.gitignore`
- Specifies which files and directories Git should ignore.

## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/wolderufael/Credit-Scoring-Model-for-eCommerce-Platform.git
   cd Credit-Scoring-Model-for-eCommerce-Platform
2. **Create and activate a virtual environment**:
   ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
3. **Install dependencies**:
   ```bash
    pip install -r requirements.txt
4. **Set environment variables**:
    * Modify the values in `.env` as needed.
5. **Run Jupyter Notebooks**:
    * You can launch the Jupyter notebooks located in the `notebook/` folder to see data processing and scraping steps.

## Contributing
1. Fork the repository.
2. Create a new feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature`').
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.
## License
This project is licensed under the terms of the MIT License. See the `LICENSE` file for details.