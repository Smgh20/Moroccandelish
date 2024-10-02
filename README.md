# Moroccandelish

## Project Overview
The **Moroccandelish** is a project that recommends traditional Moroccan recipes to users based on their ingredients, preferences, and dietary restrictions. By leveraging collaborative filtering, content-based filtering, and hybrid approaches, the system personalizes recommendations and helps users explore the rich heritage of Moroccan cuisine.

## Features
- **Personalized Recipe Recommendations**: Suggests relevant Moroccan dishes based on user input.
- **Content-based Filtering**: Uses ingredients to match user preferences with recipes.
- **Collaborative Filtering**: Recommends recipes based on similarities between users with shared tastes.
- **Interactive User Interface**: Built using Streamlit for easy user interaction.
- **Feedback System**: Collects user feedback to improve recommendations.

## Technologies Used
- **Python**: Core programming language.
- **Streamlit**: Used for building the user interface.
- **Pandas**: For data manipulation and cleaning.
- **NumPy**: Used for numerical computations.
- **TF-IDF & Cosine Similarity**: For content-based filtering.
- **Collaborative Filtering**: Implemented to enhance recommendation quality.
- **Jupyter Notebooks**: For prototyping and experimenting with models.

## How to Run the Project
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Smgh20/Moroccandelish.git
   cd Moroccandelish

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   
3. **Run the application**:
   ```bash
   streamlit run recipes.py
   
## Dataset
The system uses a custom-created dataset of Moroccan recipes. This dataset was built by scraping and compiling recipes from various sources like:
- **[Taste of Maroc](https://tasteofmaroc.com/)**
- **[Kaggle Recipe Datasets](https://www.kaggle.com/datasets)**

## Project Architecture
1. **Data Collection**: The dataset includes a wide variety of Moroccan recipes.
2. **Data Preprocessing**: Cleans the dataset by removing duplicates, handling missing values, and normalizing ingredient names.
3. **Recommendation Model**:
   - **Content-based Filtering**: Uses the TF-IDF method to calculate similarity between recipes based on ingredients.
   - **Collaborative Filtering**: Finds similarities between users based on their recipe preferences.

## Example Screenshot


## Future Work
- Integrating user profiles to store personalized preferences.
- Incorporating nutritional information to help users make healthier choices.
- Adding social media integration for recipe sharing and feedback.

