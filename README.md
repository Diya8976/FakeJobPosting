
# Fake Job Posting Detection 

This project aims to detect fake job postings using Natural Language Processing (NLP) techniques, leveraging data from a Kaggle dataset. By analyzing job posting details, the model achieves high accuracy in identifying fraudulent postings.

## Project Overview
Fake job postings have become increasingly common, leading to wasted time, resources, and frustration for job seekers. This project addresses the issue by creating a machine learning model that can effectively differentiate between legitimate and fake job postings.




## Features

- Dataset: Kaggle dataset containing job posting details.
- Techniques Used: 
    - Data cleaning and preprocessing using Pandas and NumPy.
    - Text analysis and vectorization for feature extraction.
    - Supervised learning methods for classification.
- Model Performance: Achieved 90% accuracy in detecting fake job postings.



## 🛠 Tools and Technology
- Python: Main programming language.
- Pandas and NumPy: For data manipulation and numerical computations.
- Natural Language Processing (NLP): Text analysis and feature extraction.
- Machine Learning Frameworks: Used for training and evaluating the model.

## Project Motivation
This project was inspired by my personal experience of being misled by a fake job posting, which led to a lack of promised salary and experience documentation. The aim is to create a tool that can help others avoid similar situations.
## Steps to Reproduce

1. Clone the repository:

```bash
  ngit clone https://github.com/Diya8976/FakeJobPosting.git
```
2. Install dependencies:

```bash
  pip install -r requirements.txt
```
3. Run the data preprocessing script:

```bash
  python preprocess.py
```
4. Train the model:

```bash
  python train_model.py
```
5. Evaluate the model:

```bash
  python evaluate_model.py
```

## Dataset

The dataset used in this project is available on [kaggle](https://www.kaggle.com/), containing labeled examples of real and fake job postings. For detailed information, visit the dataset page.

## Results
- Accuracy : 90%
- Key Features : The model identifies fraudulent postings by analyzing job descriptions, titles, and other textual details.

## Future Work
- Improve model accuracy by experimenting with advanced NLP techniques.
- Deploy the model as a web or mobile application for real-time use.
- Expand the dataset for better generalization across industries.

## Contributing
Contributions are welcome! If you'd like to improve the project or add features, feel free to fork the repository and submit a pull request.


## License

This project is licensed under the MIT License. See the [LICENSE](https://choosealicense.com/licenses/mit/) file for details.
## Acknowledgements

- [Kaggle](https://www.kaggle.com/) for providing the dataset.
- Open-source contributors for tools and libraries.